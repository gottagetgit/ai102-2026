"""
search_create_index.py
======================
Demonstrates provisioning an Azure AI Search index end-to-end:
  - Define an index schema with searchable, filterable, sortable, and facetable fields
  - Create a data source connection pointing to Azure Blob Storage
  - Define a skillset with built-in cognitive skills:
      * OCR (extract text from images/PDFs)
      * Language Detection
      * Entity Recognition
      * Key Phrase Extraction
  - Create an indexer that binds data-source → skillset → index
  - Run (trigger) the indexer and poll its status

AI-102 Exam Skills Mapped:
  - Provision an Azure AI Search resource, create an index, and define a skillset
  - Create data sources and indexers
  - Create and run an indexer

Required environment variables (see .env.sample):
  AZURE_SEARCH_ENDPOINT        - e.g. https://<service>.search.windows.net
  AZURE_SEARCH_ADMIN_KEY       - Admin API key (or use managed identity)
  AZURE_SEARCH_INDEX_NAME      - Name for the search index
  AZURE_STORAGE_CONNECTION_STR - Connection string for Blob Storage data source
  AZURE_STORAGE_CONTAINER_NAME - Blob container holding source documents
  AZURE_AI_SERVICES_KEY        - Cognitive Services key for enrichment skills

Package: azure-search-documents>=11.6.0
"""

import os
import time
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    # Index schema
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    CorsOptions,
    # Scoring profile
    ScoringProfile,
    TextWeights,
    # Semantic
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    # Indexer pipeline
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceType,
    SearchIndexer,
    IndexingParameters,
    IndexingParametersConfiguration,
    # Skillset
    SearchIndexerSkillset,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    # Built-in skills
    OcrSkill,
    MergeSkill,
    LanguageDetectionSkill,
    EntityRecognitionSkill,
    KeyPhraseExtractionSkill,
    # Knowledge store (referenced but detailed in search_knowledge_store.py)
    CognitiveServicesAccountKey,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ai102-demo-index")
STORAGE_CONNECTION_STR = os.environ["AZURE_STORAGE_CONNECTION_STR"]
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "documents")
AI_SERVICES_KEY = os.environ["AZURE_AI_SERVICES_KEY"]

DATA_SOURCE_NAME = f"{INDEX_NAME}-datasource"
SKILLSET_NAME = f"{INDEX_NAME}-skillset"
INDEXER_NAME = f"{INDEX_NAME}-indexer"

credential = AzureKeyCredential(SEARCH_ADMIN_KEY)


# ---------------------------------------------------------------------------
# 1. Define the index schema
# ---------------------------------------------------------------------------
def build_index() -> SearchIndex:
    """
    Build a SearchIndex with a variety of field types demonstrating:
      - SimpleField: stored but NOT searchable (IDs, booleans, numbers)
      - SearchableField: full-text searchable (analyzed by a language analyzer)
      - ComplexField: nested object or collection of objects

    Key field attributes for the AI-102 exam:
      key=True         → unique document identifier
      searchable=True  → included in full-text search
      filterable=True  → supports $filter OData expressions
      sortable=True    → supports $orderby
      facetable=True   → supports aggregation counts in faceted navigation
      retrievable=True → returned in search results
    """
    fields = [
        # Document key — must be a string, key=True
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # Full-text searchable title — uses standard analyzer by default
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            sortable=True,
            analyzer_name="en.microsoft",  # Language-aware analyzer
        ),
        # Main document content (populated by OCR + Merge skills)
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        # Metadata fields from Blob Storage
        SimpleField(
            name="metadata_storage_name",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="metadata_storage_path",
            type=SearchFieldDataType.String,
            retrievable=True,
        ),
        SimpleField(
            name="metadata_content_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # Cognitive skill outputs
        SearchableField(
            name="merged_text",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="key_phrases",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        # Entities — stored as a collection of complex objects
        ComplexField(
            name="entities",
            collection=True,
            fields=[
                SearchableField(name="text", type=SearchFieldDataType.String),
                SimpleField(
                    name="category",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="confidence",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True,
                ),
            ],
        ),
        # Numeric / date fields demonstrating filterable + sortable
        SimpleField(
            name="page_count",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SimpleField(
            name="last_modified",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        # Vector field for hybrid search (1536 dims = text-embedding-ada-002)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile",  # Defined in search_semantic_vector.py
        ),
    ]

    # Scoring profile: boost title matches 3x, content matches 1.5x
    scoring_profiles = [
        ScoringProfile(
            name="titleBoost",
            text_weights=TextWeights(weights={"title": 3.0, "content": 1.5}),
        )
    ]

    # Semantic configuration — enables semantic ranker
    semantic_config = SemanticConfiguration(
        name="default-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="merged_text")],
            keywords_fields=[SemanticField(field_name="key_phrases")],
        ),
    )

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        scoring_profiles=scoring_profiles,
        default_scoring_profile="titleBoost",
        semantic_search=SemanticSearch(configurations=[semantic_config]),
        cors_options=CorsOptions(allowed_origins=["*"], max_age_in_seconds=300),
    )


# ---------------------------------------------------------------------------
# 2. Create data source connection (Azure Blob Storage)
# ---------------------------------------------------------------------------
def build_data_source() -> SearchIndexerDataSourceConnection:
    """
    A data source tells the indexer WHERE to read documents.
    Supported types: azureblob, azuresql, cosmosdb, adlsgen2, mysql, etc.
    """
    return SearchIndexerDataSourceConnection(
        name=DATA_SOURCE_NAME,
        type=SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string=STORAGE_CONNECTION_STR,
        container=SearchIndexerDataContainer(
            name=CONTAINER_NAME,
            query=None,  # Optional: folder prefix filter, e.g. "invoices/"
        ),
        # data_change_detection_policy and data_deletion_detection_policy
        # can be added here for incremental indexing
    )


# ---------------------------------------------------------------------------
# 3. Define skillset with built-in cognitive skills
# ---------------------------------------------------------------------------
def build_skillset() -> SearchIndexerSkillset:
    """
    A skillset is an AI enrichment pipeline that transforms raw content.
    Each skill reads from /document/* context and writes enriched outputs.

    Skill execution order (implicit dependency graph):
      1. OCR         → extracts text from image regions
      2. Merge       → merges OCR text with native text into one field
      3. Language    → detects language of merged text
      4. Entity Rec. → extracts named entities (people, orgs, locations)
      5. Key Phrase  → extracts important key phrases
    """

    # Skill 1: OCR — extract text from images embedded in PDFs / images
    ocr_skill = OcrSkill(
        name="ocr-skill",
        description="Extract text from images using OCR",
        context="/document/normalized_images/*",  # Applies per image
        default_language_code="en",
        should_detect_orientation=True,
        inputs=[
            InputFieldMappingEntry(name="image", source="/document/normalized_images/*"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="text", target_name="text"),
            OutputFieldMappingEntry(name="layoutText", target_name="layoutText"),
        ],
    )

    # Skill 2: Merge — combine native text with OCR-extracted text
    merge_skill = MergeSkill(
        name="merge-skill",
        description="Merge native text and OCR text into a single field",
        context="/document",
        insert_pre_tag=" ",
        insert_post_tag=" ",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(
                name="itemsToInsert",
                source="/document/normalized_images/*/text",
            ),
            InputFieldMappingEntry(
                name="offsets",
                source="/document/normalized_images/*/contentOffset",
            ),
        ],
        outputs=[
            OutputFieldMappingEntry(name="mergedText", target_name="merged_text"),
        ],
    )

    # Skill 3: Language Detection
    language_skill = LanguageDetectionSkill(
        name="language-skill",
        description="Detect document language",
        context="/document",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/merged_text"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="languageCode", target_name="language"),
        ],
    )

    # Skill 4: Named Entity Recognition (NER)
    entity_skill = EntityRecognitionSkill(
        name="entity-skill",
        description="Extract named entities (people, orgs, locations, etc.)",
        context="/document",
        categories=["Person", "Organization", "Location", "DateTime", "URL", "Email"],
        default_language_code="en",
        include_typeless_entities=False,
        minimum_precision=0.5,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/merged_text"),
            InputFieldMappingEntry(name="languageCode", source="/document/language"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="entities", target_name="entities"),
        ],
    )

    # Skill 5: Key Phrase Extraction
    key_phrase_skill = KeyPhraseExtractionSkill(
        name="keyphrases-skill",
        description="Extract key phrases from document text",
        context="/document",
        default_language_code="en",
        maximum_key_phrase_count=20,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/merged_text"),
            InputFieldMappingEntry(name="languageCode", source="/document/language"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="keyPhrases", target_name="key_phrases"),
        ],
    )

    return SearchIndexerSkillset(
        name=SKILLSET_NAME,
        description="Built-in cognitive enrichment pipeline for AI-102 demo",
        skills=[ocr_skill, merge_skill, language_skill, entity_skill, key_phrase_skill],
        # Attach a Cognitive Services account so skills can exceed free tier
        cognitive_services_account=CognitiveServicesAccountKey(key=AI_SERVICES_KEY),
    )


# ---------------------------------------------------------------------------
# 4. Define the indexer
# ---------------------------------------------------------------------------
def build_indexer() -> SearchIndexer:
    """
    The indexer orchestrates the pipeline:
      data source → (skillset enrichment) → index

    Field mappings control how source document fields map to index fields.
    Output field mappings move skillset outputs into index fields.
    """
    return SearchIndexer(
        name=INDEXER_NAME,
        description="AI-102 demo indexer",
        data_source_name=DATA_SOURCE_NAME,
        skillset_name=SKILLSET_NAME,
        target_index_name=INDEX_NAME,
        # Source field mappings: blob metadata → index fields
        field_mappings=[
            # Map the blob URL to the document key (base64-encoded)
            # Azure Search auto-generates this mapping for blob datasources
        ],
        # Output field mappings: skillset outputs → index fields
        output_field_mappings=[
            # Merged text from merge skill
            {"sourceFieldName": "/document/merged_text", "targetFieldName": "merged_text"},
            {"sourceFieldName": "/document/language", "targetFieldName": "language"},
            {"sourceFieldName": "/document/key_phrases", "targetFieldName": "key_phrases"},
            {"sourceFieldName": "/document/entities", "targetFieldName": "entities"},
        ],
        parameters=IndexingParameters(
            batch_size=10,
            max_failed_items=5,           # Allow up to 5 failed docs before stopping
            max_failed_items_per_batch=2,
            configuration=IndexingParametersConfiguration(
                parsing_mode="default",       # Use "json" for JSON blobs, "jsonArray" for arrays
                image_action="generateNormalizedImages",  # Required for OCR
                normalized_image_max_width=2000,
                normalized_image_max_height=2000,
                indexed_file_name_extensions=".pdf,.docx,.doc,.xlsx,.pptx,.txt,.png,.jpg",
            ),
        ),
        # schedule=IndexingSchedule(interval=timedelta(hours=1))  # Uncomment for scheduled runs
    )


# ---------------------------------------------------------------------------
# Main: wire everything together
# ---------------------------------------------------------------------------
def main():
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    indexer_client = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=credential)

    try:
        # Step 1: Create or update the index
        print(f"Creating/updating index '{INDEX_NAME}'...")
        index = build_index()
        result = index_client.create_or_update_index(index)
        print(f"  Index '{result.name}' ready. Fields: {len(result.fields)}")

        # Step 2: Create or update the data source
        print(f"Creating data source '{DATA_SOURCE_NAME}'...")
        data_source = build_data_source()
        ds_result = indexer_client.create_or_update_data_source_connection(data_source)
        print(f"  Data source '{ds_result.name}' ready.")

        # Step 3: Create or update the skillset
        print(f"Creating skillset '{SKILLSET_NAME}'...")
        skillset = build_skillset()
        ss_result = indexer_client.create_or_update_skillset(skillset)
        print(f"  Skillset '{ss_result.name}' ready. Skills: {len(ss_result.skills)}")

        # Step 4: Create or update the indexer
        print(f"Creating indexer '{INDEXER_NAME}'...")
        indexer = build_indexer()
        ix_result = indexer_client.create_or_update_indexer(indexer)
        print(f"  Indexer '{ix_result.name}' ready.")

        # Step 5: Run the indexer immediately
        print("Running indexer...")
        indexer_client.run_indexer(INDEXER_NAME)
        print("  Indexer triggered. Polling status...")

        # Step 6: Poll indexer status
        for attempt in range(12):  # Poll for up to ~60 seconds
            time.sleep(5)
            status = indexer_client.get_indexer_status(INDEXER_NAME)
            last_run = status.last_result
            if last_run:
                state = last_run.status
                print(
                    f"  [{attempt+1}] Status: {state} | "
                    f"Docs succeeded: {last_run.item_count} | "
                    f"Docs failed: {last_run.failed_item_count}"
                )
                if state in ("success", "transientFailure", "persistentFailure"):
                    if last_run.errors:
                        print("  Errors:")
                        for err in last_run.errors[:3]:
                            print(f"    - {err.error_message}")
                    break
            else:
                print(f"  [{attempt+1}] Indexer running (no result yet)...")

        print("\nDone! Check the Azure portal or run search_query.py to test the index.")

    except HttpResponseError as e:
        print(f"Azure Search API error: {e.message}")
        raise
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise


if __name__ == "__main__":
    main()
