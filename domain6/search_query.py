"""
search_query.py
===============
Demonstrates querying an Azure AI Search index using the full range
of query capabilities tested on the AI-102 exam:

  1. Simple keyword search (default simple query parser)
  2. Full Lucene query syntax (boosting, proximity, regex, fuzzy)
  3. OData $filter expressions (arithmetic, string functions, geospatial)
  4. $orderby sorting (single/multi-field, ascending/descending)
  5. Wildcard searches (prefix, suffix, infix)
  6. Faceted navigation ($facets for aggregation counts)
  7. Hit highlighting ($highlight) — surround matched terms in results
  8. Autocomplete and Suggest (typeahead)
  9. Field selection ($select) to limit returned fields
 10. Pagination (skip/top for basic; search_after for deep pagination)

AI-102 Exam Skills Mapped:
  - Query an index, including syntax, sorting, filtering, and wildcards

Required environment variables (see .env.sample):
  AZURE_SEARCH_ENDPOINT      - https://<service>.search.windows.net
  AZURE_SEARCH_QUERY_KEY     - Query-only API key (least-privilege)
  AZURE_SEARCH_INDEX_NAME    - Name of the index to query

Package: azure-search-documents>=11.6.0
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryType,
    QueryAnswerType,
    QueryCaptionType,
    VectorizedQuery,
)

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_QUERY_KEY = os.environ["AZURE_SEARCH_QUERY_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ai102-demo-index")

client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_QUERY_KEY),
)


# ---------------------------------------------------------------------------
# Helper: print search results concisely
# ---------------------------------------------------------------------------
def print_results(results, label: str, fields=("id", "title", "language", "@search.score")):
    """Pretty-print search results for demo output."""
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    count = 0
    for result in results:
        count += 1
        parts = []
        for f in fields:
            val = result.get(f)
            if val is not None:
                # Truncate long values
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                parts.append(f"{f}={val_str}")
        print(f"  [{count}] {' | '.join(parts)}")

    # Print facets if present
    facets = results.get_facets()
    if facets:
        print("\n  Facets:")
        for field, buckets in facets.items():
            print(f"    {field}:")
            for bucket in buckets:
                print(f"      {bucket['value']}: {bucket['count']}")

    # Print count if present
    total = results.get_count()
    if total is not None:
        print(f"\n  Total matching documents: {total}")


# ---------------------------------------------------------------------------
# 1. Simple keyword search
# ---------------------------------------------------------------------------
def demo_simple_search():
    """
    Default query_type='simple'. Supports:
      - AND (default between terms): "azure ai" → docs with both words
      - OR:  "azure | ai"
      - NOT: "azure -openai"
      - Phrase: '"azure cognitive services"'
      - Prefix: "azure*"  (only prefix wildcards in simple mode)
    """
    results = client.search(
        search_text="azure ai services",
        query_type=QueryType.SIMPLE,
        search_mode="all",          # 'all' = AND between terms (vs 'any' = OR)
        search_fields=["title", "content", "key_phrases"],  # Limit which fields are searched
        select=["id", "title", "language"],
        top=5,
        include_total_count=True,
    )
    print_results(results, "1. Simple Keyword Search: 'azure ai services'")


# ---------------------------------------------------------------------------
# 2. Full Lucene query syntax
# ---------------------------------------------------------------------------
def demo_lucene_syntax():
    """
    query_type='full' enables the complete Lucene query syntax:

    Boosting:   title:azure^3 content:azure    (title matches worth 3x more)
    Proximity:  "azure services"~5             (terms within 5 words of each other)
    Fuzzy:      azure~1                        (Levenshtein distance 1 — handles typos)
    Regex:      /az[a-z]+/                     (regular expression)
    Field:      title:azure AND language:en
    Range:      page_count:[5 TO 50]           (inclusive range)
    Boolean:    (azure OR microsoft) AND NOT deprecated
    """
    examples = [
        # Boost title matches
        ("title:azure^3 content:machine learning", "Lucene: Field boost — title^3"),
        # Fuzzy search (typo tolerance)
        ("artificail~1", "Lucene: Fuzzy search 'artificail~1' (typo)"),
        # Proximity search
        ('"machine learning"~3', "Lucene: Proximity — 'machine learning' within 3 words"),
        # Range query
        ("page_count:[1 TO 10]", "Lucene: Range filter page_count:[1 TO 10]"),
    ]

    for query, label in examples:
        try:
            results = client.search(
                search_text=query,
                query_type=QueryType.FULL,
                select=["id", "title"],
                top=3,
                include_total_count=True,
            )
            print_results(results, f"2. {label}")
        except HttpResponseError as e:
            print(f"\n  [{label}] Error: {e.message}")


# ---------------------------------------------------------------------------
# 3. OData $filter expressions
# ---------------------------------------------------------------------------
def demo_odata_filters():
    """
    OData $filter runs BEFORE text search and uses exact field values.
    Only filterable fields can be used in $filter.

    Supported OData functions:
      String:  search.in(), startswith(), endswith(), geo.distance()
      Logic:   and, or, not
      Compare: eq, ne, lt, le, gt, ge
      Null:    field ne null
    """
    filter_examples = [
        # Equality
        ("language eq 'en'", "Filter: language eq 'en'"),
        # Combined AND
        ("language eq 'en' and page_count gt 5", "Filter: language eq 'en' AND page_count > 5"),
        # search.in() — efficient multi-value filter (replaces many ORs)
        ("search.in(language, 'en,fr,de', ',')", "Filter: search.in() for multiple languages"),
        # Null check
        ("language ne null", "Filter: exclude docs with no language"),
        # Date range
        (
            "last_modified ge 2024-01-01T00:00:00Z and last_modified lt 2025-01-01T00:00:00Z",
            "Filter: date range 2024",
        ),
    ]

    for odata_filter, label in filter_examples:
        try:
            results = client.search(
                search_text="*",           # Match all documents
                filter=odata_filter,
                select=["id", "title", "language"],
                top=3,
                include_total_count=True,
            )
            print_results(results, f"3. {label}")
        except HttpResponseError as e:
            print(f"\n  [{label}] Error: {e.message}")


# ---------------------------------------------------------------------------
# 4. Sorting with $orderby
# ---------------------------------------------------------------------------
def demo_sorting():
    """
    $orderby sorts results by one or more fields.
    Only sortable fields can be used.
    'search.score()' can be included to mix relevance with field sort.
    """
    sort_examples = [
        # Single field ascending
        (["title asc"], "Sort: title ascending"),
        # Multiple fields
        (["language asc", "page_count desc"], "Sort: language asc, page_count desc"),
        # Relevance score descending, then title
        (["search.score() desc", "title asc"], "Sort: score desc, title asc"),
    ]

    for order_by, label in sort_examples:
        try:
            results = client.search(
                search_text="azure",
                order_by=order_by,
                select=["id", "title", "language", "page_count"],
                top=3,
            )
            print_results(results, f"4. {label}", fields=("id", "title", "language", "page_count"))
        except HttpResponseError as e:
            print(f"\n  [{label}] Error: {e.message}")


# ---------------------------------------------------------------------------
# 5. Wildcard searches
# ---------------------------------------------------------------------------
def demo_wildcards():
    """
    Wildcards work in full Lucene mode only (query_type='full'):
      *  — matches zero or more characters (prefix or suffix)
      ?  — matches exactly one character

    NOTE: Leading wildcards (*azure) are expensive — disabled by default.
    Enable in index config: allowLeadingWildcard on the field.

    Simple mode supports only prefix wildcards via trailing *:
      "azure*"  → matches azure, azureml, azure-openai, etc.
    """
    wildcard_examples = [
        # Prefix wildcard (simple mode)
        ("azure*", QueryType.SIMPLE, "Wildcard: prefix 'azure*' (simple)"),
        # Suffix wildcard (full Lucene)
        ("*learning", QueryType.FULL, "Wildcard: suffix '*learning' (Lucene)"),
        # Single-char wildcard (full Lucene)
        ("az?re", QueryType.FULL, "Wildcard: single-char 'az?re' (Lucene)"),
        # Infix wildcard (full Lucene)
        ("mac*ine", QueryType.FULL, "Wildcard: infix 'mac*ine' (Lucene)"),
    ]

    for query, qtype, label in wildcard_examples:
        try:
            results = client.search(
                search_text=query,
                query_type=qtype,
                select=["id", "title"],
                top=3,
            )
            print_results(results, f"5. {label}")
        except HttpResponseError as e:
            print(f"\n  [{label}] Error: {e.message}")


# ---------------------------------------------------------------------------
# 6. Faceted navigation
# ---------------------------------------------------------------------------
def demo_facets():
    """
    Facets return aggregation counts for filterable fields —
    used to build navigation filters ("filter by category", etc.).

    Syntax: "field,count:N,sort:count"
    Options:
      count:N   — return top N facet values (default 10)
      sort:value — sort facets alphabetically (default: sort by count desc)
      interval:N — for numeric fields, group into buckets of size N
    """
    results = client.search(
        search_text="*",
        facets=[
            "language,count:10",
            "metadata_content_type,count:5",
            "page_count,interval:10",     # Numeric intervals: 0-9, 10-19, etc.
            "entities/category,count:5",  # Facet on nested field
        ],
        select=["id", "title"],
        top=0,          # We only want facets, not actual documents
        include_total_count=True,
    )
    print_results(results, "6. Faceted Navigation")


# ---------------------------------------------------------------------------
# 7. Hit highlighting
# ---------------------------------------------------------------------------
def demo_highlighting():
    """
    Highlighting wraps matched terms in search results with HTML tags
    (default: <em>...</em>) so the UI can display them bold/highlighted.

    highlight_pre_tag / highlight_post_tag control the wrapping tags.
    highlight_post_tag defaults to </em>.
    """
    results = client.search(
        search_text="machine learning azure",
        highlight_fields="content,title",
        highlight_pre_tag='<strong class="highlight">',
        highlight_post_tag="</strong>",
        select=["id", "title"],
        top=3,
    )
    print(f"\n{'='*60}")
    print(" 7. Hit Highlighting")
    print(f"{'='*60}")
    for result in results:
        print(f"\n  Document: {result.get('title', result.get('id', 'N/A'))}")
        highlights = result.get("@search.highlights", {})
        for field, snippets in highlights.items():
            print(f"  Highlights in '{field}':")
            for snippet in snippets:
                print(f"    ... {snippet} ...")


# ---------------------------------------------------------------------------
# 8. Autocomplete and Suggest
# ---------------------------------------------------------------------------
def demo_autocomplete_suggest():
    """
    Autocomplete and Suggest require a suggester defined in the index schema.
    Add this to your SearchIndex:
        suggesters=[SearchSuggester(name="sg", source_fields=["title", "content"])]

    Autocomplete: completes the CURRENT partial term (single field)
    Suggest: returns full DOCUMENT suggestions matching the partial query
    """
    from azure.search.documents import SearchClient

    # Autocomplete: what terms start with "mach"?
    print(f"\n{'='*60}")
    print(" 8a. Autocomplete (partial term: 'mach')")
    print(f"{'='*60}")
    try:
        auto_results = client.autocomplete(
            search_text="mach",
            suggester_name="sg",
            mode="oneTerm",     # 'oneTerm' or 'twoTerms' or 'oneTermWithContext'
            top=5,
        )
        for item in auto_results:
            print(f"  Autocomplete: '{item['text']}' (query: '{item['query_plus_text']}')")
    except HttpResponseError as e:
        print(f"  Autocomplete requires a suggester: {e.message}")

    # Suggest: full document suggestions for "mach"
    print(f"\n{'='*60}")
    print(" 8b. Suggest (partial: 'mach')")
    print(f"{'='*60}")
    try:
        suggest_results = client.suggest(
            search_text="mach",
            suggester_name="sg",
            select=["id", "title"],
            top=5,
            order_by=["search.score() desc"],
        )
        for item in suggest_results:
            print(f"  Suggestion: {item.get('title', 'N/A')} (score: {item.get('@search.score')})")
    except HttpResponseError as e:
        print(f"  Suggest requires a suggester: {e.message}")


# ---------------------------------------------------------------------------
# 9. Field selection and pagination
# ---------------------------------------------------------------------------
def demo_pagination():
    """
    Pagination strategies:

    Basic (skip/top):
      - Simple but costly at high page numbers (skip=1000 still processes 1000 docs)
      - Max skip is 100,000

    Deep pagination (search_after — recommended for large result sets):
      - Uses a continuation token from the last result
      - More efficient for paging through thousands of results
    """
    print(f"\n{'='*60}")
    print(" 9. Pagination — Page 1 (top=3, skip=0)")
    print(f"{'='*60}")

    # Page 1
    results = client.search(
        search_text="azure",
        top=3,
        skip=0,
        select=["id", "title"],
        order_by=["title asc"],    # Consistent order required for pagination
        include_total_count=True,
    )
    docs = list(results)
    total = results.get_count()
    print(f"  Page 1 of {total} total results:")
    for doc in docs:
        print(f"    {doc.get('title', doc.get('id'))}")

    # Page 2
    print(f"\n{'='*60}")
    print(" 9. Pagination — Page 2 (top=3, skip=3)")
    print(f"{'='*60}")
    results2 = client.search(
        search_text="azure",
        top=3,
        skip=3,
        select=["id", "title"],
        order_by=["title asc"],
    )
    for doc in results2:
        print(f"    {doc.get('title', doc.get('id'))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Azure AI Search — Query Demonstrations")
    print("Index:", INDEX_NAME)

    try:
        demo_simple_search()
        demo_lucene_syntax()
        demo_odata_filters()
        demo_sorting()
        demo_wildcards()
        demo_facets()
        demo_highlighting()
        demo_autocomplete_suggest()
        demo_pagination()
        print("\nAll query demos complete!")

    except HttpResponseError as e:
        print(f"\nAzure Search API error [{e.status_code}]: {e.message}")
        raise
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise


if __name__ == "__main__":
    main()
