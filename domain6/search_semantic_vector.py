"""
search_semantic_vector.py
=========================
Demonstrates semantic ranking and vector (hybrid) search in Azure AI Search:

  1. Update an existing index to add:
       - A vector field (content_vector) for dense embeddings
       - HNSW vector search configuration
       - Semantic configuration (semantic ranker)

  2. Generate embeddings using Azure OpenAI text-embedding-3-small

  3. Upload documents with pre-computed embeddings

  4. Run query types:
       a. Pure vector search (knn over embedding field)
       b. BM25 full-text search (baseline)
       c. Hybrid search (BM25 + vector — reciprocal rank fusion)
       d. Hybrid + semantic reranker (best quality)
       e. Semantic captions and answers

AI-102 Exam Skills Mapped:
  - Implement semantic and vector store solutions

Key concepts:
  - Vector search uses HNSW (Hierarchical Navigable Small World) graphs
    for approximate nearest-neighbor (ANN) similarity search
  - Semantic ranker is a separate re-ranking model (L2 reranker) that reads
    the top N BM25/vector results and reorders using transformer models
  - Hybrid search combines BM25 and vector scores via Reciprocal Rank Fusion
  - Query vectorizer: auto-vectorize queries at search time

Required environment variables (see .env.sample):
  AZURE_SEARCH_ENDPOINT         - https://<service>.search.windows.net
  AZURE_SEARCH_ADMIN_KEY        - Admin API key
  AZURE_SEARCH_QUERY_KEY        - Query-only API key
  AZURE_SEARCH_INDEX_NAME       - Name of the index
  AZURE_OPENAI_ENDPOINT         - https://<resource>.openai.azure.com/
  AZURE_OPENAI_API_KEY          - Azure OpenAI API key
  AZURE_OPENAI_EMBEDDING_DEPLOY - Deployment name for embedding model

Package: azure-search-documents>=11.6.0, openai>=1.0.0
"""

import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    # Vector search config
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    # Query vectorizer — auto-embed query text at search time
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    # Semantic search config
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryAnswerType,
    QueryCaptionType,
    SemanticErrorMode,
)
from openai import AzureOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
SEARCH_QUERY_KEY = os.environ["AZURE_SEARCH_QUERY_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ai102-demo-index")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOY", "text-embedding-3-small")
EMBEDDING_DIMS = 1536  # text-embedding-3-small default; text-embedding-ada-002 = 1536

admin_credential = AzureKeyCredential(SEARCH_ADMIN_KEY)
query_credential = AzureKeyCredential(SEARCH_QUERY_KEY)

# Azure OpenAI client for generating embeddings
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version="2024-02-01",
)


# ---------------------------------------------------------------------------
# 1. Build index with vector + semantic config
# ---------------------------------------------------------------------------
def build_vector_semantic_index() -> SearchIndex:
    """
    Creates an index with:
      - A vector field using HNSW algorithm
      - A query vectorizer that auto-embeds query strings via Azure OpenAI
      - A semantic configuration for L2 re-ranking
    """

    # HNSW (Hierarchical Navigable Small World) — the standard ANN algorithm
    # Parameters:
    #   m         : max bi-directional links per node (affects recall vs. memory)
    #   ef_construction: nodes explored during index build (higher = better recall)
    #   ef_search : nodes explored during query (higher = better recall, slower)
    #   metric    : cosine (default, best for text embeddings), dotProduct, euclidean
    hnsw_config = HnswAlgorithmConfiguration(
        name="hnsw-config",
        parameters=HnswParameters(
            m=4,
            ef_construction=400,
            ef_search=500,
            metric="cosine",
        ),
    )

    # Query vectorizer: automatically embed the query string at search time
    # This means you can pass plain text to vector queries (no pre-computation)
    query_vectorizer = AzureOpenAIVectorizer(
        vectorizer_name="openai-vectorizer",
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=AOAI_ENDPOINT,
            deployment_name=EMBEDDING_DEPLOYMENT,
            model_name=EMBEDDING_DEPLOYMENT,
            api_key=AOAI_API_KEY,
        ),
    )

    # Tie algorithm + vectorizer into a named profile
    vector_profile = VectorSearchProfile(
        name="hnsw-profile",
        algorithm_configuration_name="hnsw-config",
        vectorizer_name="openai-vectorizer",
    )

    vector_search = VectorSearch(
        algorithms=[hnsw_config],
        profiles=[vector_profile],
        vectorizers=[query_vectorizer],
    )

    # Semantic configuration: which fields the semantic ranker reads
    semantic_config = SemanticConfiguration(
        name="default-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="key_phrases")],
        ),
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, sortable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(
            name="key_phrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        # Vector field: dimension MUST match the embedding model output
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )


# ---------------------------------------------------------------------------
# 2. Generate embeddings with Azure OpenAI
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """
    Generate a dense vector embedding using Azure OpenAI.
    The resulting vector is used for semantic similarity search.

    text-embedding-3-small: 1536 dimensions, fast, cost-effective
    text-embedding-3-large: 3072 dimensions, higher quality
    text-embedding-ada-002: 1536 dimensions, older model
    """
    # Strip and truncate — embedding models have token limits (~8192 for ada/3-small)
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * EMBEDDING_DIMS

    response = aoai_client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# 3. Upload sample documents with embeddings
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Introduction to Azure Machine Learning",
        "content": (
            "Azure Machine Learning is a cloud service for accelerating and managing "
            "the machine learning project lifecycle. Use it to train, deploy, and manage "
            "ML models at scale."
        ),
        "key_phrases": ["machine learning", "Azure ML", "model training", "deployment"],
        "category": "machine-learning",
    },
    {
        "id": "doc-002",
        "title": "Azure OpenAI Service Overview",
        "content": (
            "Azure OpenAI Service provides REST API access to OpenAI's powerful language models "
            "including GPT-4, DALL-E, and Whisper. Use it to build intelligent applications "
            "with natural language understanding and generation capabilities."
        ),
        "key_phrases": ["GPT-4", "language models", "REST API", "natural language"],
        "category": "ai-services",
    },
    {
        "id": "doc-003",
        "title": "Azure AI Search Vector Capabilities",
        "content": (
            "Azure AI Search supports vector search using approximate nearest neighbor algorithms. "
            "Combine BM25 keyword search with dense vector retrieval for hybrid search that improves "
            "relevance, especially for conceptual and semantic queries."
        ),
        "key_phrases": ["vector search", "HNSW", "hybrid search", "BM25", "embeddings"],
        "category": "search",
    },
    {
        "id": "doc-004",
        "title": "Azure Kubernetes Service",
        "content": (
            "Azure Kubernetes Service (AKS) simplifies deploying and managing containerized "
            "applications using Kubernetes. AKS handles health monitoring and maintenance "
            "of nodes and reduces operational overhead."
        ),
        "key_phrases": ["Kubernetes", "containers", "AKS", "orchestration"],
        "category": "infrastructure",
    },
]


def upload_documents_with_embeddings():
    """Generate embeddings and upload documents to the index."""
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=admin_credential)

    # Create/update the index
    print("Creating/updating vector+semantic index...")
    index = build_vector_semantic_index()
    index_client.create_or_update_index(index)
    print(f"  Index '{INDEX_NAME}' ready.")

    # Generate embeddings and attach to documents
    print("Generating embeddings...")
    docs_with_vectors = []
    for doc in SAMPLE_DOCUMENTS:
        text_to_embed = f"{doc['title']}. {doc['content']}"
        embedding = get_embedding(text_to_embed)
        doc_copy = dict(doc)
        doc_copy["content_vector"] = embedding
        docs_with_vectors.append(doc_copy)
        print(f"  Embedded: '{doc['title'][:50]}'")

    # Upload to index
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=admin_credential,
    )
    results = search_client.upload_documents(documents=docs_with_vectors)
    succeeded = sum(1 for r in results if r.succeeded)
    print(f"  Uploaded {succeeded}/{len(docs_with_vectors)} documents.")


# ---------------------------------------------------------------------------
# 4a. Pure vector search
# ---------------------------------------------------------------------------
def demo_vector_search(query: str):
    """
    Pure vector search: embed the query and find k nearest neighbors.
    Best for conceptual/semantic similarity — e.g., 'neural network frameworks'
    finds ML docs even if those exact words don't appear.
    """
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=query_credential,
    )

    # Option A: Pass pre-computed vector (VectorizedQuery)
    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,       # Return top 3 nearest neighbors
        fields="content_vector",     # Which vector field to search
        exhaustive=False,            # False = use HNSW (ANN); True = exact kNN
    )

    results = search_client.search(
        search_text=None,            # None = pure vector, no BM25
        vector_queries=[vector_query],
        select=["id", "title", "category"],
        top=3,
    )

    print(f"\n{'='*60}")
    print(f" 4a. Pure Vector Search: '{query}'")
    print(f"{'='*60}")
    for result in results:
        print(
            f"  [{result.get('@search.score', 0):.4f}] "
            f"{result.get('title')} ({result.get('category')})"
        )


# ---------------------------------------------------------------------------
# 4b. Hybrid search (BM25 + vector)
# ---------------------------------------------------------------------------
def demo_hybrid_search(query: str):
    """
    Hybrid search: run BM25 keyword search AND vector search in parallel,
    then merge rankings using Reciprocal Rank Fusion (RRF).

    RRF formula: score = Σ 1/(k + rank_i)  where k=60 by default
    This is robust — doesn't require score normalization between BM25 and vectors.

    Hybrid search typically outperforms either pure BM25 or pure vector search.
    """
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=query_credential,
    )

    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=50,      # Retrieve more candidates for RRF merging
        fields="content_vector",
    )

    results = search_client.search(
        search_text=query,           # BM25 component
        vector_queries=[vector_query],  # Vector component
        select=["id", "title", "category"],
        top=3,
    )

    print(f"\n{'='*60}")
    print(f" 4b. Hybrid Search (BM25 + Vector): '{query}'")
    print(f"{'='*60}")
    for result in results:
        print(
            f"  [{result.get('@search.score', 0):.4f}] "
            f"{result.get('title')} ({result.get('category')})"
        )


# ---------------------------------------------------------------------------
# 4c. Hybrid + semantic reranker
# ---------------------------------------------------------------------------
def demo_hybrid_semantic_search(query: str):
    """
    Full pipeline: BM25 + Vector → RRF merge → Semantic L2 reranker

    The semantic ranker:
      1. Reads the top 50 hybrid results
      2. Re-scores using a cross-encoder transformer model
      3. Returns semantic_score (0–4) alongside reranked results
      4. Optionally generates captions (highlighted relevant passages)
         and answers (direct extractive answers to question queries)

    Requires the 'semantic-ranker' tier — Semantic Plan on Azure AI Search.
    """
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=query_credential,
    )

    query_vector = get_embedding(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=50,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        # Semantic ranking parameters
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="default-semantic",
        # Captions: extract highlighted passages from content
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_caption_highlight_enabled=True,
        # Answers: attempt to extract a direct answer (for question queries)
        query_answer=QueryAnswerType.EXTRACTIVE,
        query_answer_count=1,
        # Handle semantic failures gracefully (fall back to hybrid score)
        semantic_error_mode=SemanticErrorMode.PARTIAL,
        semantic_max_wait_in_milliseconds=2000,
        select=["id", "title", "content", "category"],
        top=3,
    )

    print(f"\n{'='*60}")
    print(f" 4c. Hybrid + Semantic Reranker: '{query}'")
    print(f"{'='*60}")

    # Print semantic answers (direct answer to the question)
    semantic_answers = results.get_answers()
    if semantic_answers:
        print("  Semantic Answers:")
        for answer in semantic_answers:
            print(f"    Score: {answer.score:.3f}")
            print(f"    Answer: {answer.text[:200]}")

    for result in results:
        rerank_score = result.get("@search.reranker_score", "N/A")
        hybrid_score = result.get("@search.score", 0)
        print(
            f"\n  Title: {result.get('title')}"
            f"\n  Hybrid Score: {hybrid_score:.4f} | Reranker Score: {rerank_score}"
        )

        # Print captions (highlighted relevant passages)
        captions = result.get("@search.captions", [])
        if captions:
            print(f"  Caption: {captions[0].text[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Azure AI Search — Semantic & Vector Search Demonstrations")
    print(f"Index: {INDEX_NAME} | Embedding model: {EMBEDDING_DEPLOYMENT}")

    try:
        # First: set up the index and upload sample data
        print("\n--- Setup ---")
        upload_documents_with_embeddings()

        # Pause briefly to let indexing complete
        import time
        print("Waiting for indexing to complete...")
        time.sleep(3)

        # Run search demos
        test_query = "how to train and deploy AI models in the cloud"

        demo_vector_search(test_query)
        demo_hybrid_search(test_query)
        demo_hybrid_semantic_search(test_query)

        # Additional query to show conceptual matching
        conceptual_query = "container orchestration"
        demo_hybrid_search(conceptual_query)

        print("\nAll semantic/vector demos complete!")

    except HttpResponseError as e:
        print(f"\nAzure Search error [{e.status_code}]: {e.message}")
        raise
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise


if __name__ == "__main__":
    main()
