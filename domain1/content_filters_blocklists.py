"""
content_filters_blocklists.py
==============================
Demonstrates creating and managing custom blocklists with Azure AI Content Safety.
Custom blocklists let you extend the built-in harm detection with your own
domain-specific terms (e.g. competitor names, profanity, internal policy violations).

Exam Skill: "Implement responsible AI, including content filters and blocklists"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Creating a custom blocklist
  - Adding block items (exact match and regex patterns)
  - Listing and retrieving block items
  - Analyzing text against a blocklist
  - Deleting block items and blocklists (cleanup)
  - Understanding when to use blocklists vs. built-in harm categories

Key concepts:
  - Blocklists are custom word/phrase filters that work ALONGSIDE the built-in
    harm category analysis (hate, violence, self-harm, sexual)
  - Blocklist items can be exact strings or regex patterns
  - If any blocklist item matches, the API returns a match with the blocklist name
    and the matched item ID - your app decides the final action

Required packages:
  pip install azure-ai-contentsafety azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT  - e.g. https://<name>.cognitiveservices.azure.com/
  AZURE_AI_SERVICES_KEY       - API key for the resource
"""

import os
import time
from dotenv import load_dotenv
from azure.ai.contentsafety import BlocklistClient
from azure.ai.contentsafety.models import (
    TextBlocklist,
    AddOrUpdateTextBlocklistItemsOptions,
    TextBlocklistItem,
    AnalyzeTextOptions,
    TextCategory,
)
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
KEY      = os.environ["AZURE_AI_SERVICES_KEY"]

# Demo blocklist name - will be created and cleaned up
BLOCKLIST_NAME = "ai102-demo-blocklist"


def get_blocklist_client() -> BlocklistClient:
    """Create an authenticated BlocklistClient."""
    return BlocklistClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def get_content_safety_client() -> ContentSafetyClient:
    """Create an authenticated ContentSafetyClient (for text analysis)."""
    return ContentSafetyClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def create_blocklist(client: BlocklistClient, name: str, description: str) -> TextBlocklist:
    """
    Create a new custom blocklist or update an existing one.
    Blocklists are identified by name (alphanumeric + hyphens, max 64 chars).
    """
    print(f"\n[CREATE BLOCKLIST] '{name}'")
    blocklist = client.create_or_update_text_blocklist(
        blocklist_name=name,
        options=TextBlocklist(
            blocklist_name=name,
            description=description,
        ),
    )
    print(f"  Created: {blocklist.blocklist_name}")
    print(f"  Description: {blocklist.description}")
    return blocklist


def add_blocklist_items(client: BlocklistClient, blocklist_name: str) -> list:
    """
    Add terms to the blocklist.
    Each item can be:
      - An exact text string (case-insensitive by default)
      - A regex pattern (for flexible matching)

    Returns the list of created item IDs.
    """
    print(f"\n[ADD ITEMS] Adding items to blocklist '{blocklist_name}'")

    items_to_add = [
        TextBlocklistItem(
            text="badword1",
            description="Example banned term #1",
        ),
        TextBlocklistItem(
            text="competitor_xyz",
            description="Competitor name - not allowed in support chats",
        ),
        TextBlocklistItem(
            text="internal_codename_alpha",
            description="Internal project codename - must not be disclosed",
        ),
        # Regex pattern: matches phone numbers like 555-1234 or (555) 123-4567
        TextBlocklistItem(
            text=r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            description="Phone number pattern - PII protection",
        ),
    ]

    response = client.add_or_update_blocklist_items(
        blocklist_name=blocklist_name,
        options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=items_to_add),
    )

    item_ids = []
    for item in response.blocklist_items:
        print(f"  Added item: '{item.text}' (ID: {item.blocklist_item_id})")
        item_ids.append(item.blocklist_item_id)

    return item_ids


def list_blocklist_items(client: BlocklistClient, blocklist_name: str) -> None:
    """
    List all items in a blocklist.
    Useful for auditing what terms are currently blocked.
    """
    print(f"\n[LIST ITEMS] Items in blocklist '{blocklist_name}':")
    items = list(client.list_text_blocklist_items(blocklist_name=blocklist_name))
    if not items:
        print("  No items found.")
        return
    for item in items:
        print(f"  ID: {item.blocklist_item_id[:8]}... | Text: '{item.text}' | Desc: {item.description}")


def list_all_blocklists(client: BlocklistClient) -> None:
    """List all blocklists in this Content Safety resource."""
    print("\n[LIST BLOCKLISTS] All blocklists:")
    blocklists = list(client.list_text_blocklists())
    if not blocklists:
        print("  No blocklists found.")
        return
    for bl in blocklists:
        print(f"  - {bl.blocklist_name}: {bl.description}")


def analyze_text_with_blocklist(
    safety_client: ContentSafetyClient,
    text: str,
    blocklist_names: list,
) -> dict:
    """
    Analyze text using both built-in harm categories AND a custom blocklist.
    The blocklist check runs alongside the standard harm category analysis.

    If a blocklist item matches:
      - response.blocklists_match contains the match details
      - Your app should treat this as a block action
    """
    print(f"\n[ANALYZE WITH BLOCKLIST] Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    print(f"  Blocklists applied: {blocklist_names}")

    request = AnalyzeTextOptions(
        text=text,
        categories=[TextCategory.HATE, TextCategory.VIOLENCE, TextCategory.SELF_HARM, TextCategory.SEXUAL],
        blocklist_names=blocklist_names,     # Apply our custom blocklist
        halt_on_blocklist_hit=False,         # Continue harm category analysis even if blocklist matches
    )

    try:
        response = safety_client.analyze_text(request)
    except HttpResponseError as e:
        print(f"  [ERROR] Analysis failed: {e.message}")
        return {}

    # Check harm categories
    print(f"\n  Harm Category Results:")
    any_harm = False
    for cat in response.categories_analysis:
        if cat.severity > 0:
            any_harm = True
            print(f"    {cat.category}: severity={cat.severity}")
    if not any_harm:
        print("    No built-in harm categories triggered.")

    # Check blocklist matches
    print(f"\n  Blocklist Match Results:")
    if response.blocklists_match:
        for match in response.blocklists_match:
            print(f"    MATCH! Blocklist: '{match.blocklist_name}' | Item: '{match.blocklist_item_text}' | ID: {match.blocklist_item_id}")
        action = "BLOCK (blocklist match)"
    else:
        print("    No blocklist matches.")
        action = "ALLOW" if not any_harm else "REVIEW/BLOCK (harm detected)"

    print(f"\n  --> Decision: {action}")
    return {
        "harm_categories": {c.category: c.severity for c in response.categories_analysis},
        "blocklist_matches": [m.blocklist_item_text for m in (response.blocklists_match or [])],
        "action": action,
    }


def remove_blocklist_item(
    client: BlocklistClient,
    blocklist_name: str,
    item_id: str,
) -> None:
    """Remove a single item from a blocklist by its ID."""
    print(f"\n[REMOVE ITEM] Removing item {item_id[:8]}... from '{blocklist_name}'")
    try:
        client.remove_blocklist_items(
            blocklist_name=blocklist_name,
            options={"blocklist_item_ids": [item_id]},
        )
        print("  Item removed successfully.")
    except HttpResponseError as e:
        print(f"  [ERROR] Could not remove item: {e.message}")


def delete_blocklist(client: BlocklistClient, blocklist_name: str) -> None:
    """
    Delete an entire blocklist and all its items.
    Use with caution - this is irreversible.
    """
    print(f"\n[DELETE BLOCKLIST] Deleting '{blocklist_name}' ...")
    try:
        client.delete_text_blocklist(blocklist_name=blocklist_name)
        print("  Blocklist deleted.")
    except ResourceNotFoundError:
        print("  Blocklist not found (already deleted?).")
    except HttpResponseError as e:
        print(f"  [ERROR] Could not delete blocklist: {e.message}")


def main():
    print("=" * 60)
    print("Azure AI Content Safety - Custom Blocklists Demo")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT}")

    try:
        blocklist_client = get_blocklist_client()
        safety_client    = get_content_safety_client()

        # 1. Create a new blocklist
        create_blocklist(
            blocklist_client,
            BLOCKLIST_NAME,
            "AI-102 demo blocklist for exam preparation",
        )

        # 2. Add terms to the blocklist
        item_ids = add_blocklist_items(blocklist_client, BLOCKLIST_NAME)

        # Small pause to allow replication
        print("\n  [Waiting 2s for blocklist replication...]")
        time.sleep(2)

        # 3. List blocklist items to verify
        list_blocklist_items(blocklist_client, BLOCKLIST_NAME)

        # 4. List all blocklists in the resource
        list_all_blocklists(blocklist_client)

        # 5. Analyze texts with the blocklist
        test_texts = [
            "This is a completely normal sentence with no issues.",
            "Please contact our competitor_xyz support team for help.",
            "Hello, my phone number is 555-867-5309, please call me.",
            "The internal_codename_alpha project is confidential.",
        ]

        print("\n--- ANALYZING TEXTS WITH BLOCKLIST ---")
        for text in test_texts:
            analyze_text_with_blocklist(safety_client, text, [BLOCKLIST_NAME])

        # 6. Remove one item to demonstrate item management
        if item_ids:
            remove_blocklist_item(blocklist_client, BLOCKLIST_NAME, item_ids[0])
            list_blocklist_items(blocklist_client, BLOCKLIST_NAME)

        # 7. Cleanup - delete the demo blocklist
        print("\n--- CLEANUP ---")
        delete_blocklist(blocklist_client, BLOCKLIST_NAME)

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except HttpResponseError as e:
        print(f"\n[ERROR] Content Safety API error: {e.message}")
        print("Check that your endpoint and key are correct.")


if __name__ == "__main__":
    main()
