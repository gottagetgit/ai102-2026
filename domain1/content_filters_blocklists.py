"""
content_filters_blocklists.py
=============================
Demonstrates creating custom blocklists and filtering content using
Azure AI Content Safety's blocklist APIs.

Exam Skill: "Implement responsible AI - content filters and blocklists"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Creating a custom blocklist
  - Adding exact-match terms to a blocklist
  - Adding regex patterns to a blocklist
  - Analyzing text against a blocklist
  - Removing items from a blocklist
  - Deleting a blocklist
  - When to use blocklists vs. content filters

Content Safety filter vs. blocklist:
  - Filters: Built-in harm categories (hate, violence, self-harm, sexual)
    with severity levels. Configurable thresholds.
  - Blocklists: Custom word lists or regex patterns you define.
    Triggered on exact match (or regex match). Use for brand terms,
    competitor names, proprietary info, or domain-specific terms.

Required packages:
  pip install azure-ai-contentsafety python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT - your Azure AI Services endpoint
  AZURE_AI_SERVICES_KEY      - your Azure AI Services API key
"""

import os
import time
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AddOrUpdateTextBlocklistItemsOptions,
    RemoveTextBlocklistItemsOptions,
    TextBlocklistItem,
    AnalyzeTextOptions,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
KEY      = os.environ["AZURE_AI_SERVICES_KEY"]

# Blocklist name for this demo
BLOCKLIST_NAME = "ai102-demo-blocklist"


def create_client() -> ContentSafetyClient:
    """Initialize the Content Safety client."""
    return ContentSafetyClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


# ---------------------------------------------------------------------------
# Step 1: Create or update a blocklist
# ---------------------------------------------------------------------------

def create_blocklist(client: ContentSafetyClient) -> None:
    """Create a new blocklist (or update description if it already exists)."""
    print("\n--- Creating Blocklist ---")

    try:
        result = client.create_or_update_text_blocklist(
            blocklist_name=BLOCKLIST_NAME,
            options={"description": "AI-102 demo blocklist for exam prep"},
        )
        print(f"  Blocklist created: '{result.blocklist_name}'")
        print(f"  Description: {result.description}")
    except HttpResponseError as e:
        print(f"  [ERROR] Failed to create blocklist: {e.message}")
        raise


# ---------------------------------------------------------------------------
# Step 2: Add terms to the blocklist
# ---------------------------------------------------------------------------

def add_blocklist_items(client: ContentSafetyClient) -> list:
    """
    Add both exact-match terms and regex patterns to the blocklist.
    Returns list of item IDs for later use.
    """
    print("\n--- Adding Items to Blocklist ---")

    items_to_add = [
        TextBlocklistItem(
            text="contoso-secret",
            description="Exact match: proprietary term",
        ),
        TextBlocklistItem(
            text="competitor_brand_x",
            description="Exact match: competitor mention",
        ),
        TextBlocklistItem(
            text=r"\b(badword|offensive_term)\b",
            description="Regex: custom offensive terms (case-insensitive)",
        ),
        TextBlocklistItem(
            text=r"\d{3}-\d{2}-\d{4}",  # SSN pattern
            description="Regex: US Social Security Number pattern",
        ),
    ]

    try:
        result = client.add_or_update_text_blocklist_items(
            blocklist_name=BLOCKLIST_NAME,
            options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=items_to_add),
        )

        item_ids = []
        for item in result.blocklist_items:
            print(f"  Added item ID={item.blocklist_item_id}: '{item.text}'")
            item_ids.append(item.blocklist_item_id)

        print(f"  Total items added: {len(item_ids)}")
        return item_ids

    except HttpResponseError as e:
        print(f"  [ERROR] Failed to add items: {e.message}")
        raise


# ---------------------------------------------------------------------------
# Step 3: List blocklist items
# ---------------------------------------------------------------------------

def list_blocklist_items(client: ContentSafetyClient) -> None:
    """List all items in the blocklist."""
    print("\n--- Listing Blocklist Items ---")

    try:
        items = client.list_text_blocklist_items(blocklist_name=BLOCKLIST_NAME)
        for item in items:
            print(f"  ID: {item.blocklist_item_id}")
            print(f"    Text: {item.text}")
            print(f"    Description: {item.description}")
    except HttpResponseError as e:
        print(f"  [ERROR] Failed to list items: {e.message}")


# ---------------------------------------------------------------------------
# Step 4: Analyze text with blocklist
# ---------------------------------------------------------------------------

def analyze_with_blocklist(client: ContentSafetyClient) -> None:
    """
    Analyze text against the custom blocklist.
    The blocklist_names parameter enables blocklist matching.
    """
    print("\n--- Analyzing Text with Blocklist ---")

    test_texts = [
        "Hello, this is a normal message with no issues.",
        "Please don't share our contoso-secret with anyone outside the company.",
        "The competitor_brand_x product is inferior to ours.",
        "User SSN is 123-45-6789, please handle with care.",
        "This message has badword in it.",
    ]

    for text in test_texts:
        print(f"\n  Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        try:
            request = AnalyzeTextOptions(
                text=text,
                blocklist_names=[BLOCKLIST_NAME],  # Enable our blocklist
                halt_on_blocklist_hit=False,       # Continue analysis even if blocked
            )
            result = client.analyze_text(request)

            if result.blocklists_match:
                print(f"  BLOCKED - Blocklist matches:")
                for match in result.blocklists_match:
                    print(f"    - Blocklist: {match.blocklist_name}")
                    print(f"      Item ID: {match.blocklist_item_id}")
                    print(f"      Matched text: '{match.blocklist_item_text}'")
            else:
                print(f"  OK - No blocklist matches")

            # Also show content safety categories
            if result.categories_analysis:
                flagged = [(c.category, c.severity) for c in result.categories_analysis if c.severity > 0]
                if flagged:
                    print(f"  Content categories flagged: {flagged}")

        except HttpResponseError as e:
            print(f"  [ERROR] Analysis failed: {e.message}")


# ---------------------------------------------------------------------------
# Step 5: Remove specific items
# ---------------------------------------------------------------------------

def remove_blocklist_items(client: ContentSafetyClient, item_ids: list) -> None:
    """Remove specific items from the blocklist by ID."""
    print("\n--- Removing Items from Blocklist ---")

    if not item_ids:
        print("  No item IDs to remove")
        return

    # Remove just the first item as a demo
    ids_to_remove = item_ids[:1]

    try:
        client.remove_text_blocklist_items(
            blocklist_name=BLOCKLIST_NAME,
            options=RemoveTextBlocklistItemsOptions(blocklist_item_ids=ids_to_remove),
        )
        print(f"  Removed {len(ids_to_remove)} item(s): {ids_to_remove}")
    except HttpResponseError as e:
        print(f"  [ERROR] Failed to remove items: {e.message}")


# ---------------------------------------------------------------------------
# Step 6: Clean up - delete the blocklist
# ---------------------------------------------------------------------------

def delete_blocklist(client: ContentSafetyClient) -> None:
    """Delete the entire blocklist and all its items."""
    print("\n--- Deleting Blocklist ---")

    try:
        client.delete_text_blocklist(blocklist_name=BLOCKLIST_NAME)
        print(f"  Blocklist '{BLOCKLIST_NAME}' deleted successfully")
    except HttpResponseError as e:
        print(f"  [ERROR] Failed to delete blocklist: {e.message}")


# ---------------------------------------------------------------------------
# Bonus: List all blocklists
# ---------------------------------------------------------------------------

def list_all_blocklists(client: ContentSafetyClient) -> None:
    """List all blocklists in the Content Safety resource."""
    print("\n--- Listing All Blocklists ---")

    try:
        blocklists = client.list_text_blocklists()
        found = False
        for bl in blocklists:
            found = True
            print(f"  Blocklist: '{bl.blocklist_name}'")
            print(f"    Description: {bl.description}")
        if not found:
            print("  No blocklists found")
    except HttpResponseError as e:
        print(f"  [ERROR] Failed to list blocklists: {e.message}")


if __name__ == "__main__":
    print("=" * 60)
    print("Content Safety Blocklists Demo")
    print("=" * 60)

    client = create_client()

    # Show existing blocklists before starting
    list_all_blocklists(client)

    # Create blocklist
    create_blocklist(client)

    # Add items
    item_ids = add_blocklist_items(client)

    # Wait a moment for propagation
    print("\n  Waiting for blocklist to propagate...")
    time.sleep(3)

    # List items to confirm
    list_blocklist_items(client)

    # Analyze text
    analyze_with_blocklist(client)

    # Remove an item
    remove_blocklist_items(client, item_ids)

    # Final cleanup
    delete_blocklist(client)

    # Confirm it's gone
    list_all_blocklists(client)

    print("\n" + "=" * 60)
    print("Blocklist Demo Complete")
    print("=" * 60)
