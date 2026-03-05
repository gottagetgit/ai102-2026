"""
search_knowledge_store.py
=========================
Demonstrates configuring Knowledge Store projections in an Azure AI Search skillset.

A Knowledge Store persists AI enrichment outputs to Azure Storage so they can be
consumed by tools like Power BI, Azure Data Factory, or custom analytics pipelines —
independently of the search index.

Three projection types:
  1. Table projections  → Azure Table Storage (structured, queryable, great for Power BI)
  2. Object projections → Azure Blob Storage as JSON files (full enriched document tree)
  3. File projections   → Azure Blob Storage as raw binary files (normalized images from OCR)

How projections work:
  - Each projection references a "shaper" skill output or inline shaping expressions
  - Projections are defined in projection GROUPS; within a group, projections share
    the same document root, enabling cross-referencing in Power BI
  - The 'generatedKeyAsId' field becomes the partition/row key for table projections

AI-102 Exam Skills Mapped:
  - Manage Knowledge Store projections, including file, object, and table projections

Required environment variables (see .env.sample):
  AZURE_SEARCH_ENDPOINT        - https://<service>.search.windows.net
  AZURE_SEARCH_ADMIN_KEY       - Admin API key
  AZURE_SEARCH_INDEX_NAME      - Existing index / skillset name
  AZURE_STORAGE_CONNECTION_STR - Storage account for knowledge store output
  AZURE_AI_SERVICES_KEY        - Cognitive Services multi-service key

Package: azure-search-documents>=11.6.0
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerSkillset,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    # Built-in skills
    OcrSkill,
    MergeSkill,
    LanguageDetectionSkill,
    EntityRecognitionSkill,
    KeyPhraseExtractionSkill,
    # Shaper skill: reshapes enrichment tree for projection
    ShaperSkill,
    # Knowledge store
    SearchIndexerKnowledgeStore,
    SearchIndexerKnowledgeStoreProjection,
    SearchIndexerKnowledgeStoreTableProjectionSelector,
    SearchIndexerKnowledgeStoreObjectProjectionSelector,
    SearchIndexerKnowledgeStoreFileProjectionSelector,
    CognitiveServicesAccountKey,
)

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ai102-demo-index")
STORAGE_CONNECTION_STR = os.environ["AZURE_STORAGE_CONNECTION_STR"]
AI_SERVICES_KEY = os.environ["AZURE_AI_SERVICES_KEY"]

SKILLSET_NAME = f"{INDEX_NAME}-ks-skillset"   # Separate skillset for knowledge store demo

credential = AzureKeyCredential(SEARCH_ADMIN_KEY)


# ---------------------------------------------------------------------------
# Build a skillset with Knowledge Store projections
# ---------------------------------------------------------------------------
def build_skillset_with_knowledge_store() -> SearchIndexerSkillset:
    """
    Creates a skillset that enriches documents AND persists enrichment output
    to an Azure Storage Knowledge Store in three projection formats.

    The ShaperSkill is the key pattern:
      - It consolidates enrichment outputs into a single clean object
      - That object is then referenced in table/object/file projections
      - Inline shaping (via /document/... paths) works too, but ShaperSkill
        is more readable and reusable
    """

    # -----------------------------------------------------------------------
    # Enrichment skills (same as search_create_index.py but standalone)
    # -----------------------------------------------------------------------
    ocr_skill = OcrSkill(
        name="ocr",
        context="/document/normalized_images/*",
        default_language_code="en",
        should_detect_orientation=True,
        inputs=[InputFieldMappingEntry(name="image", source="/document/normalized_images/*")],
        outputs=[
            OutputFieldMappingEntry(name="text", target_name="text"),
        ],
    )

    merge_skill = MergeSkill(
        name="merge",
        context="/document",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
            InputFieldMappingEntry(name="itemsToInsert", source="/document/normalized_images/*/text"),
            InputFieldMappingEntry(name="offsets", source="/document/normalized_images/*/contentOffset"),
        ],
        outputs=[OutputFieldMappingEntry(name="mergedText", target_name="merged_text")],
    )

    language_skill = LanguageDetectionSkill(
        name="language",
        context="/document",
        inputs=[InputFieldMappingEntry(name="text", source="/document/merged_text")],
        outputs=[OutputFieldMappingEntry(name="languageCode", target_name="language")],
    )

    entity_skill = EntityRecognitionSkill(
        name="entities",
        context="/document",
        categories=["Person", "Organization", "Location", "DateTime"],
        default_language_code="en",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/merged_text"),
            InputFieldMappingEntry(name="languageCode", source="/document/language"),
        ],
        outputs=[OutputFieldMappingEntry(name="entities", target_name="entities")],
    )

    keyphrases_skill = KeyPhraseExtractionSkill(
        name="keyphrases",
        context="/document",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/merged_text"),
            InputFieldMappingEntry(name="languageCode", source="/document/language"),
        ],
        outputs=[OutputFieldMappingEntry(name="keyPhrases", target_name="key_phrases")],
    )

    # -----------------------------------------------------------------------
    # ShaperSkill: create a structured object for table projection
    #
    # The output is a /document/tableprojection node containing exactly the
    # fields you want in the 'Documents' table in Knowledge Store.
    # -----------------------------------------------------------------------
    shaper_skill = ShaperSkill(
        name="shaper",
        description="Shape enrichments for Knowledge Store table projection",
        context="/document",
        inputs=[
            InputFieldMappingEntry(
                name="document_id",
                source="/document/metadata_storage_path",
            ),
            InputFieldMappingEntry(
                name="document_name",
                source="/document/metadata_storage_name",
            ),
            InputFieldMappingEntry(
                name="content_type",
                source="/document/metadata_content_type",
            ),
            InputFieldMappingEntry(
                name="language",
                source="/document/language",
            ),
            InputFieldMappingEntry(
                name="merged_text",
                source="/document/merged_text",
            ),
            InputFieldMappingEntry(
                name="key_phrases",
                source="/document/key_phrases",
            ),
            # Nested: entities become a sub-table in the Knowledge Store
            InputFieldMappingEntry(
                name="entities",
                source="/document/entities",
            ),
        ],
        outputs=[
            OutputFieldMappingEntry(
                name="output",
                target_name="tableprojection",
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Entity-level shaper: projects entities into their own row-level table
    # This creates a separate "Entities" table linked to the Documents table
    # via the generatedKeyAsId
    # -----------------------------------------------------------------------
    entity_shaper = ShaperSkill(
        name="entity-shaper",
        description="Shape individual entities for their own table projection",
        context="/document/entities/*",   # Runs once PER entity (collection)
        inputs=[
            InputFieldMappingEntry(name="entity_text", source="/document/entities/*/text"),
            InputFieldMappingEntry(name="category", source="/document/entities/*/category"),
            InputFieldMappingEntry(name="confidence", source="/document/entities/*/confidenceScore"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="output", target_name="entity_row"),
        ],
    )

    # -----------------------------------------------------------------------
    # Knowledge Store: define projection groups
    #
    # A projection GROUP is a set of table+object+file projections that share
    # the same enrichment root. Use multiple groups to produce different
    # "views" of your data.
    # -----------------------------------------------------------------------

    # GROUP 1: Structured projections — tables + a full JSON object blob
    group_1 = SearchIndexerKnowledgeStoreProjection(
        tables=[
            # Main documents table — one row per document
            SearchIndexerKnowledgeStoreTableProjectionSelector(
                table_name="Documents",
                generated_key_as_id="document_id",    # Auto-generated join key
                source="/document/tableprojection",
            ),
            # Key phrases table — one row per key phrase per document
            SearchIndexerKnowledgeStoreTableProjectionSelector(
                table_name="KeyPhrases",
                generated_key_as_id="keyphrase_id",
                source="/document/tableprojection/key_phrases/*",
            ),
            # Entities table — one row per entity per document
            SearchIndexerKnowledgeStoreTableProjectionSelector(
                table_name="Entities",
                generated_key_as_id="entity_id",
                source="/document/entities/*",
            ),
        ],
        objects=[
            # Full enriched document as JSON blob — easy to inspect/debug
            SearchIndexerKnowledgeStoreObjectProjectionSelector(
                storage_container="knowledge-store-objects",
                generated_key_as_id="document_id",
                source="/document/tableprojection",
            ),
        ],
        files=[],   # File projections in Group 2
    )

    # GROUP 2: File projections — normalized images extracted from PDFs
    group_2 = SearchIndexerKnowledgeStoreProjection(
        tables=[],
        objects=[],
        files=[
            # Each normalized_image is stored as a separate image file in blob
            SearchIndexerKnowledgeStoreFileProjectionSelector(
                storage_container="knowledge-store-images",
                generated_key_as_id="image_id",
                source="/document/normalized_images/*",
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Knowledge Store configuration
    # -----------------------------------------------------------------------
    knowledge_store = SearchIndexerKnowledgeStore(
        storage_connection_string=STORAGE_CONNECTION_STR,
        projections=[group_1, group_2],
    )

    return SearchIndexerSkillset(
        name=SKILLSET_NAME,
        description="Skillset with Knowledge Store projections for AI-102 demo",
        skills=[
            ocr_skill,
            merge_skill,
            language_skill,
            entity_skill,
            keyphrases_skill,
            shaper_skill,
            entity_shaper,
        ],
        cognitive_services_account=CognitiveServicesAccountKey(key=AI_SERVICES_KEY),
        knowledge_store=knowledge_store,
    )


# ---------------------------------------------------------------------------
# Inspect what was projected to the Knowledge Store
# ---------------------------------------------------------------------------
def inspect_knowledge_store_output():
    """
    After the indexer runs, the Knowledge Store tables will be in Azure Table Storage
    and objects/files will be in Azure Blob Storage.

    This function lists what was projected using the Azure Storage SDK.
    """
    try:
        from azure.data.tables import TableServiceClient
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        print("Install azure-data-tables and azure-storage-blob to inspect output:")
        print("  pip install azure-data-tables azure-storage-blob")
        return

    print("\n--- Inspecting Knowledge Store Output ---")

    # Table Storage: list entities in the Documents table
    try:
        table_service = TableServiceClient.from_connection_string(STORAGE_CONNECTION_STR)
        table_client = table_service.get_table_client("Documents")
        entities = list(table_client.list_entities())
        print(f"\nDocuments table: {len(entities)} rows")
        for entity in entities[:3]:
            print(f"  PartitionKey={entity.get('PartitionKey')} | "
                  f"language={entity.get('language')} | "
                  f"content_type={entity.get('content_type')}")
    except Exception as e:
        print(f"  Could not read Table Storage: {e}")

    # Blob Storage: list projected JSON objects
    try:
        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STR)
        container_client = blob_service.get_container_client("knowledge-store-objects")
        blobs = list(container_client.list_blobs())
        print(f"\nObject projections (JSON): {len(blobs)} blobs")
        for blob in blobs[:3]:
            print(f"  {blob.name} ({blob.size} bytes)")
    except Exception as e:
        print(f"  Could not read Blob Storage (objects): {e}")

    # Blob Storage: list projected image files
    try:
        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STR)
        img_container = blob_service.get_container_client("knowledge-store-images")
        images = list(img_container.list_blobs())
        print(f"\nFile projections (images): {len(images)} files")
        for img in images[:3]:
            print(f"  {img.name} ({img.size} bytes)")
    except Exception as e:
        print(f"  Could not read Blob Storage (images): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    indexer_client = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=credential)

    try:
        print(f"Creating skillset '{SKILLSET_NAME}' with Knowledge Store projections...")
        skillset = build_skillset_with_knowledge_store()
        result = indexer_client.create_or_update_skillset(skillset)

        print(f"  Skillset '{result.name}' created. Skills: {len(result.skills)}")
        print(f"  Knowledge Store: {result.knowledge_store is not None}")
        print(f"  Projection groups: {len(result.knowledge_store.projections)}")

        for i, group in enumerate(result.knowledge_store.projections):
            table_count = len(group.tables or [])
            object_count = len(group.objects or [])
            file_count = len(group.files or [])
            print(
                f"    Group {i+1}: {table_count} table(s), "
                f"{object_count} object(s), {file_count} file projection(s)"
            )

        print("\nProjection tables that will be created in Azure Table Storage:")
        print("  - Documents (one row per document)")
        print("  - KeyPhrases (one row per key phrase, linked to Documents)")
        print("  - Entities (one row per entity, linked to Documents)")
        print("\nBlob containers:")
        print("  - knowledge-store-objects/ (full JSON enrichment per document)")
        print("  - knowledge-store-images/  (OCR-normalized images per document)")

        print("\nNext steps:")
        print("  1. Create/reset an indexer that references this skillset")
        print("  2. Run the indexer")
        print("  3. Open Azure Storage Explorer or Power BI to explore the Knowledge Store")
        print("  4. Run this script with --inspect to list projected data")

        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
            inspect_knowledge_store_output()

    except HttpResponseError as e:
        print(f"Azure Search API error: {e.message}")
        raise
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise


if __name__ == "__main__":
    main()
