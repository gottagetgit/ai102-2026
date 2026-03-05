"""
rag_pattern.py
==============
Demonstrates a complete Retrieval-Augmented Generation (RAG) pattern:
  1. Embed the user query using Azure OpenAI text embeddings
  2. Search Azure AI Search (vector search) for relevant documents
  3. Pass retrieved documents as context to a chat completion

Exam Skill: "Implement a RAG pattern by grounding a model in your data"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Creating embeddings with Azure OpenAI (text-embedding-ada-002 / 3-small / 3-large)
  - Uploading documents to Azure AI Search with vector fields
  - Performing hybrid search (keyword + vector) 
  - Building a prompt with retrieved context
  - Azure OpenAI's built-in "on your data" feature (data_sources parameter)
  - Measuring relevance through cosine similarity

RAG pipeline overview:
  [User Query]
      ↓ embed query
  [Embedding Vector]
      ↓ vector search
  [Azure AI Search] → [Top-K relevant chunks]
      ↓ inject into prompt
  [Chat Completion] → [Grounded Answer]

Required packages:
  pip install openai azure-search-documents azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT           - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY                - API key
  AZURE_OPENAI_DEPLOYMENT         - Chat deployment e.g. "gpt-4o"
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT - Embedding deployment e.g. "text-embedding-3-small"
  AZURE_SEARCH_ENDPOINT           - e.g. https://<name>.search.windows.net
  AZURE_SEARCH_KEY                - Admin or query key
  AZURE_SEARCH_INDEX              - Index name e.g. "ai102-docs"
"""

import os
import json
import math
from dotenv import load_dotenv
import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

load_dotenv()

OPENAI_ENDPOINT    = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY         = os.environ["AZURE_OPENAI_KEY"]
CHAT_DEPLOYMENT    = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
EMBED_DEPLOYMENT   = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
SEARCH_ENDPOINT    = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY         = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX       = os.environ.get("AZURE_SEARCH_INDEX", "ai102-rag-demo")

# Embedding dimensions (must match your embedding model):
#   text-embedding-ada-002  : 1536
#   text-embedding-3-small  : 1536 (default), supports 256-1536 with dimensions param
#   text-embedding-3-large  : 3072 (default), supports 256-3072 with dimensions param
EMBEDDING_DIMENSIONS = 1536


def get_openai_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview",
    )


def get_search_clients() -> tuple[SearchIndexClient, SearchClient]:
    """Return both a SearchIndexClient (admin) and SearchClient (query)."""
    credential     = AzureKeyCredential(SEARCH_KEY)
    index_client   = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    search_client  = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX, credential=credential)
    return index_client, search_client


# ---------------------------------------------------------------------------
# Step 1: Create embedding for text
# ---------------------------------------------------------------------------

def embed_text(openai_client: openai.AzureOpenAI, text: str) -> list[float]:
    """
    Create a dense vector embedding for the given text.

    Embeddings are numerical representations of text in high-dimensional space.
    Semantically similar texts will have similar (close) vector representations.
    This is what enables semantic/vector search.
    """
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_DEPLOYMENT,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Range: -1 (opposite) to 1 (identical). Values > 0.85 indicate high similarity.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Step 2: Setup Azure AI Search index with vector field
# ---------------------------------------------------------------------------

def create_search_index(index_client: SearchIndexClient) -> None:
    """
    Create an Azure AI Search index with:
      - id         : Unique document identifier
      - title      : Searchable text field
      - content    : Main document text (keyword searchable)
      - embedding  : Vector field for semantic search
    """
    print(f"\n[SEARCH INDEX] Creating index '{SEARCH_INDEX}'...")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                # HNSW (Hierarchical Navigable Small World) is the recommended
                # algorithm for most use cases - good balance of speed/accuracy
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    index = SearchIndex(
        name=SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
    )

    try:
        index_client.create_or_update_index(index)
        print(f"  Index '{SEARCH_INDEX}' created/updated.")
    except HttpResponseError as e:
        print(f"  [ERROR] Could not create index: {e.message}")
        raise


# ---------------------------------------------------------------------------
# Step 3: Upload documents with embeddings
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "id": "1",
        "title": "Azure AI Content Safety Overview",
        "content": (
            "Azure AI Content Safety is a service that detects harmful content in text and images. "
            "It analyzes content across four harm categories: hate speech, violence, self-harm, and sexual content. "
            "Each category has a severity score from 0 to 6. Severity 0 means safe, 2 is low risk, "
            "4 is medium risk, and 6 is high risk. Organizations can set thresholds for each category "
            "based on their use case and audience."
        ),
    },
    {
        "id": "2",
        "title": "Azure OpenAI Service Models",
        "content": (
            "Azure OpenAI Service provides access to advanced language models including GPT-4o, GPT-4, "
            "and GPT-3.5 Turbo. GPT-4o is a multimodal model that can process both text and images. "
            "For image generation, DALL-E 3 is available. For semantic search and embeddings, "
            "text-embedding-3-small and text-embedding-3-large models are available with dimensions "
            "ranging from 256 to 3072."
        ),
    },
    {
        "id": "3",
        "title": "Azure AI Search Vector Search",
        "content": (
            "Azure AI Search supports vector search using HNSW (Hierarchical Navigable Small World) "
            "algorithm for approximate nearest neighbor search. You can perform pure vector search, "
            "pure keyword search, or hybrid search combining both. Hybrid search uses Reciprocal Rank "
            "Fusion (RRF) to merge results from both search types. Semantic ranking can further rerank "
            "results using language models."
        ),
    },
    {
        "id": "4",
        "title": "RAG Pattern Best Practices",
        "content": (
            "When implementing Retrieval-Augmented Generation (RAG), key considerations include: "
            "chunk size (typically 512-1024 tokens), chunk overlap (10-20% for context continuity), "
            "and retrieval count (top 3-5 chunks). Always cite sources in the generated response. "
            "Use hybrid search for better recall. Implement prompt shields to prevent injection attacks "
            "in retrieved content. Monitor retrieval quality with metrics like NDCG and MRR."
        ),
    },
    {
        "id": "5",
        "title": "Azure AI Services Authentication",
        "content": (
            "Azure AI Services support two authentication methods: API key authentication and "
            "Microsoft Entra ID (formerly Azure Active Directory). For production workloads, "
            "Entra ID with managed identity is recommended as it eliminates key management. "
            "Use DefaultAzureCredential from azure-identity which automatically selects the "
            "appropriate credential source: AzureCliCredential for local development, "
            "ManagedIdentityCredential for Azure-hosted services."
        ),
    },
]


def upload_documents(
    openai_client: openai.AzureOpenAI,
    search_client: SearchClient,
) -> None:
    """
    Generate embeddings for each document and upload to Azure AI Search.
    In production, this indexing step runs once (or on document updates),
    not on every query.
    """
    print(f"\n[UPLOAD] Embedding and uploading {len(SAMPLE_DOCUMENTS)} documents...")

    documents_with_embeddings = []
    for doc in SAMPLE_DOCUMENTS:
        # Embed the content (or title + content for better recall)
        text_to_embed = f"{doc['title']}. {doc['content']}"
        embedding = embed_text(openai_client, text_to_embed)
        documents_with_embeddings.append({**doc, "embedding": embedding})
        print(f"  Embedded: {doc['title'][:50]}")

    result = search_client.upload_documents(documents=documents_with_embeddings)
    print(f"  Uploaded {len(result)} documents.")
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"  Succeeded: {succeeded}/{len(result)}")


# ---------------------------------------------------------------------------
# Step 4: Retrieve relevant documents
# ---------------------------------------------------------------------------

def retrieve_documents(
    openai_client: openai.AzureOpenAI,
    search_client: SearchClient,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Embed the query and perform hybrid search (vector + keyword) to
    retrieve the most relevant documents.

    Hybrid search advantages:
      - Vector search finds semantically similar content even with different keywords
      - Keyword search finds exact term matches that vector search might miss
      - Combined via Reciprocal Rank Fusion (RRF) for best of both worlds
    """
    print(f"\n[RETRIEVE] Query: '{query}'")

    # Embed the query
    query_vector = embed_text(openai_client, query)

    # Hybrid search: both text and vector
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = list(search_client.search(
        search_text=query,           # Keyword search
        vector_queries=[vector_query], # Vector search
        select=["id", "title", "content"],
        top=top_k,
    ))

    print(f"  Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        score = doc.get("@search.score", 0)
        print(f"    {i}. [{score:.4f}] {doc['title']}")

    return results


# ---------------------------------------------------------------------------
# Step 5: Generate answer with retrieved context
# ---------------------------------------------------------------------------

def generate_grounded_answer(
    openai_client: openai.AzureOpenAI,
    query: str,
    retrieved_docs: list[dict],
) -> str:
    """
    Generate an answer using the retrieved documents as context.
    This is the 'G' (Generation) step in RAG.

    Key patterns:
      - Inject context before the user question
      - Instruct the model to use ONLY the provided context
      - Ask the model to cite sources
      - Instruct it to say "I don't know" when context is insufficient
    """
    print(f"\n[GENERATE] Building grounded answer...")

    # Format retrieved documents as context
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[Source {i}] {doc['title']}:\n{doc['content']}")
    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context.
Rules:
- Answer using only the information in the context below
- Cite your sources using [Source N] notation
- If the context doesn't contain enough information, say "I don't have enough information to answer that."
- Do not add information from your training data"""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=400,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    print(f"\n  Answer:\n{answer}")
    print(f"\n  Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
    return answer


# ---------------------------------------------------------------------------
# Optional: Azure OpenAI native "on your data" feature
# ---------------------------------------------------------------------------

def demo_openai_on_your_data(openai_client: openai.AzureOpenAI, query: str) -> None:
    """
    Azure OpenAI has a built-in 'on your data' feature that handles the
    retrieval step automatically. You configure the data source and the API
    handles embedding, search, and prompt construction.

    This is simpler than manual RAG but less customizable.
    """
    print(f"\n[ON YOUR DATA] Using Azure OpenAI's native data source integration")
    print(f"  Query: {query}")

    try:
        response = openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": query}],
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": SEARCH_ENDPOINT,
                            "index_name": SEARCH_INDEX,
                            "authentication": {
                                "type": "api_key",
                                "key": SEARCH_KEY,
                            },
                            "query_type": "vector_simple_hybrid",
                            "embedding_dependency": {
                                "type": "deployment_name",
                                "deployment_name": EMBED_DEPLOYMENT,
                            },
                            "top_n_documents": 3,
                            "in_scope": True,
                        },
                    }
                ]
            },
            max_tokens=400,
        )
        answer = response.choices[0].message.content
        print(f"  Answer:\n{answer}")
    except Exception as e:
        print(f"  [INFO] On-your-data feature not available or configured: {e}")
        print("  This feature requires the index to be set up with the Azure portal or REST API.")


def main():
    print("=" * 60)
    print("Azure OpenAI + AI Search RAG Pattern Demo")
    print("=" * 60)

    try:
        openai_client               = get_openai_client()
        index_client, search_client = get_search_clients()

        # Step 1: Create the search index
        create_search_index(index_client)

        # Step 2: Embed and upload documents
        upload_documents(openai_client, search_client)

        # Step 3: Run RAG queries
        queries = [
            "What severity levels does Azure AI Content Safety use?",
            "How does authentication work for Azure AI Services?",
            "What are best practices for chunking in RAG systems?",
        ]

        for query in queries:
            print("\n" + "=" * 60)
            # Retrieve relevant docs
            docs = retrieve_documents(openai_client, search_client, query, top_k=3)
            # Generate grounded answer
            generate_grounded_answer(openai_client, query, docs)

        # Optional: Azure OpenAI native integration
        demo_openai_on_your_data(openai_client, queries[0])

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
        print("Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY")
    except openai.APIError as e:
        print(f"\n[ERROR] OpenAI error: {e}")
    except HttpResponseError as e:
        print(f"\n[ERROR] Azure Search error: {e.message}")


if __name__ == "__main__":
    main()
