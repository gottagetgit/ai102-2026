# Domain 6: Knowledge Mining and Information Extraction Solutions

This directory contains Python demos for **Domain 6** of the AI-102 exam:
**Implement knowledge mining and information extraction solutions (15–20%)**.

---

## Files

| File | Purpose | Exam Skill |
|------|---------|------------|
| `search_create_index.py` | Provision an AI Search index, skillset, data source, and indexer | Provision resource, create index, define skillset, create and run indexer |
| `search_custom_skill.py` | Implement a custom skill with the WebApiSkill contract | Implement custom skills and include them in a skillset |
| `search_query.py` | Full-text search, OData filters, sorting, wildcards, facets, highlighting | Query an index: syntax, sorting, filtering, wildcards |
| `search_semantic_vector.py` | Semantic ranker, HNSW vector search, hybrid BM25+vector search | Implement semantic and vector store solutions |
| `search_knowledge_store.py` | Knowledge Store table, object, and file projections | Manage Knowledge Store projections |
| `document_intelligence_prebuilt.py` | Invoice, receipt, layout, and read prebuilt models | Provision Document Intelligence, use prebuilt models |
| `document_intelligence_custom.py` | Train, test, and compose custom Document Intelligence models | Implement, train, test, and publish custom models; create composed models |
| `content_understanding.py` | Multi-modal analysis: documents, images, video, audio via Content Understanding | OCR pipeline, summarize/classify documents, extract entities/tables, multi-modal ingestion |

---

## Exam Skills Coverage

### Azure AI Search
- **Provision an Azure AI Search resource** — `search_create_index.py`
- **Create an index** — `search_create_index.py` (field types, analyzers, scoring profiles)
- **Define a skillset** — `search_create_index.py` (OCR, language, entity, key phrase)
- **Create data sources and indexers** — `search_create_index.py`
- **Create and run an indexer** — `search_create_index.py` (trigger + poll status)
- **Implement custom skills** — `search_custom_skill.py` (WebApiSkill contract)
- **Query an index** — `search_query.py` (simple, Lucene, OData, wildcards, facets, highlights)
- **Semantic and vector store solutions** — `search_semantic_vector.py`
- **Knowledge Store projections** — `search_knowledge_store.py` (tables, objects, files)

### Azure Document Intelligence
- **Provision Document Intelligence** — `document_intelligence_prebuilt.py`
- **Use prebuilt models** — `document_intelligence_prebuilt.py` (invoice, receipt, layout, read)
- **Implement a custom model** — `document_intelligence_custom.py`
- **Train, test, publish** — `document_intelligence_custom.py`
- **Create a composed model** — `document_intelligence_custom.py`

### Azure Content Understanding
- **OCR pipeline** — `content_understanding.py` (document OCR analyzer)
- **Summarize/classify documents** — `content_understanding.py` (LLM-based fields)
- **Extract entities and tables** — `content_understanding.py` (entity extraction analyzer)
- **Multi-modal processing** — `content_understanding.py` (document, image, video, audio)

---

## Required Environment Variables

All files use `python-dotenv` to load from a `.env` file in the repo root.
See `../.env.sample` for the full template.

| Variable | Used By | Description |
|----------|---------|-------------|
| `AZURE_SEARCH_ENDPOINT` | All search scripts | `https://<service>.search.windows.net` |
| `AZURE_SEARCH_ADMIN_KEY` | `search_create_index`, `search_semantic_vector`, `search_knowledge_store`, `search_custom_skill` | Admin API key (write access) |
| `AZURE_SEARCH_QUERY_KEY` | `search_query`, `search_semantic_vector` | Query-only key (least privilege) |
| `AZURE_SEARCH_INDEX_NAME` | All search scripts | Name of the demo index |
| `AZURE_STORAGE_CONNECTION_STR` | `search_create_index`, `search_knowledge_store` | Blob Storage connection string |
| `AZURE_STORAGE_CONTAINER_NAME` | `search_create_index` | Blob container with source documents |
| `AZURE_AI_SERVICES_KEY` | `search_create_index`, `search_knowledge_store`, `content_understanding` | Cognitive Services multi-service key |
| `AZURE_OPENAI_ENDPOINT` | `search_semantic_vector` | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | `search_semantic_vector` | Azure OpenAI API key |
| `AZURE_OPENAI_EMBEDDING_DEPLOY` | `search_semantic_vector` | Embedding deployment name (e.g. `text-embedding-3-small`) |
| `AZURE_CUSTOM_SKILL_URL` | `search_custom_skill` | HTTPS URL of deployed Azure Function skill |
| `AZURE_FUNCTION_KEY` | `search_custom_skill` | Azure Function host key (optional) |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | `document_intelligence_prebuilt`, `document_intelligence_custom` | DI resource endpoint |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | `document_intelligence_prebuilt`, `document_intelligence_custom` | DI API key |
| `AZURE_TRAINING_CONTAINER_SAS_URL` | `document_intelligence_custom` | SAS URL to labeled training data container |
| `AZURE_CUSTOM_MODEL_ID` | `document_intelligence_custom` | Custom model ID to test |
| `AZURE_CUSTOM_MODEL_ID_2` | `document_intelligence_custom` | Second custom model for compose demo |
| `AZURE_COMPOSED_MODEL_ID` | `document_intelligence_custom` | Composed model ID |
| `AZURE_AI_SERVICES_ENDPOINT` | `content_understanding` | Content Understanding endpoint |

---

## Running the Scripts

```bash
# Install dependencies
pip install -r ../requirements.txt

# Copy and fill in your environment variables
cp ../.env.sample ../.env
# Edit .env with your Azure resource values

# Run individual demos
python search_create_index.py       # Creates index, skillset, indexer
python search_query.py              # Runs all query types
python search_semantic_vector.py    # Hybrid + semantic search
python search_knowledge_store.py    # Configures Knowledge Store projections
python search_custom_skill.py       # Tests custom skill locally
python document_intelligence_prebuilt.py  # Prebuilt model demos
python document_intelligence_custom.py    # Custom model lifecycle
python content_understanding.py     # Multi-modal content analysis
```

### Custom Skill Local Test
```bash
# Start the local skill server (no Azure required)
python search_custom_skill.py --serve

# Register the skill in your Azure Search skillset
python search_custom_skill.py --register
```

---

## Key Concepts for the Exam

### Azure AI Search Index Field Attributes
| Attribute | Effect |
|-----------|--------|
| `key=True` | Unique document identifier (required, string) |
| `searchable=True` | Included in full-text search (analyzed) |
| `filterable=True` | Supports `$filter` OData expressions |
| `sortable=True` | Supports `$orderby` |
| `facetable=True` | Supports aggregation counts |
| `retrievable=True` | Returned in search results |

### Query Types
| Type | Use Case |
|------|---------|
| `simple` | Default; AND/OR/NOT, phrase, prefix wildcard |
| `full` | Lucene: boost, proximity, fuzzy, regex, range |
| `semantic` | Re-ranks top results with transformer model |

### Document Intelligence Models
| Model | Purpose |
|-------|---------|
| `prebuilt-read` | OCR only — fastest, most general |
| `prebuilt-layout` | Structure: tables, paragraphs, selection marks |
| `prebuilt-invoice` | Invoice fields: vendor, total, line items |
| `prebuilt-receipt` | Receipt fields: merchant, items, total |
| `custom neural` | Varied layouts; recommended for most custom cases |
| `custom template` | Fixed-layout forms; highest precision on exact layouts |
| Composed | Routes to best-matching component model |

### Content Understanding Scenarios
| Scenario | Use Case |
|----------|---------|
| `documentIntelligence` | OCR + layout extraction |
| `contentExtraction` | LLM-based field extraction |
| `imageAnalysis` | Visual content understanding |
| `videoContentUnderstanding` | Transcript, scenes, chapters |
| `audioContentUnderstanding` | Transcription + meeting insights |
