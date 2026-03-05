"""
create_ai_resource.py
=====================
Demonstrates creating and managing Azure AI Services resources
using the Azure Management SDK.

Exam Skill: "Create an Azure AI resource"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Creating a multi-service Azure AI Services resource
  - Creating a single-service resource (Azure OpenAI)
  - Listing all AI resources in a resource group
  - Getting resource details (endpoint, properties)
  - Understanding resource kinds and SKU tiers
  - Handling resource existence checks (idempotent creation)

Resource kinds:
  - CognitiveServices  = multi-service AI (Language, Vision, Speech, etc.)
  - OpenAI             = Azure OpenAI Service
  - TextAnalytics      = Language service (single)
  - ComputerVision     = Vision service (single)
  - SpeechServices     = Speech service (single)
  - FormRecognizer     = Document Intelligence (single)
  - ContentSafety      = Content Safety (single)

SKU tiers:
  - F0  = Free tier (limited requests/month, not for production)
  - S0  = Standard tier (pay-per-use, production-ready)

Required packages:
  pip install azure-identity azure-mgmt-cognitiveservices python-dotenv

Required environment variables (in .env):
  AZURE_SUBSCRIPTION_ID  - your Azure subscription ID
  AZURE_RESOURCE_GROUP   - resource group name
  AZURE_AI_ACCOUNT_NAME  - name for the AI Services account
  AZURE_LOCATION         - Azure region (e.g., 'eastus')
"""

import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import (
    Account,
    AccountProperties,
    Sku,
)
from azure.core.exceptions import HttpResponseError, ResourceExistsError

load_dotenv()

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP  = os.environ["AZURE_RESOURCE_GROUP"]
ACCOUNT_NAME    = os.environ["AZURE_AI_ACCOUNT_NAME"]
LOCATION        = os.environ.get("AZURE_LOCATION", "eastus")


def create_management_client() -> CognitiveServicesManagementClient:
    """
    Create the Cognitive Services management client.
    Uses DefaultAzureCredential (requires az login or service principal).
    """
    credential = DefaultAzureCredential()
    return CognitiveServicesManagementClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
    )


# ---------------------------------------------------------------------------
# Create a multi-service AI resource
# ---------------------------------------------------------------------------

def create_multi_service_resource(client: CognitiveServicesManagementClient) -> None:
    """
    Create an Azure AI Services multi-service resource.
    Kind='CognitiveServices' gives access to Language, Vision, Speech, etc.
    This is the recommended approach for most AI-102 scenarios.
    """
    print("\n--- Creating Multi-Service AI Resource ---")
    print(f"  Name: {ACCOUNT_NAME}")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Location: {LOCATION}")
    print(f"  Kind: CognitiveServices (multi-service)")
    print(f"  SKU: S0 (Standard)")

    account = Account(
        location=LOCATION,
        kind="CognitiveServices",          # Multi-service resource
        sku=Sku(name="S0"),               # Standard tier
        properties=AccountProperties(
            custom_sub_domain_name=ACCOUNT_NAME.lower().replace("_", "-"),
        ),
    )

    try:
        poller = client.accounts.begin_create(
            resource_group_name=RESOURCE_GROUP,
            account_name=ACCOUNT_NAME,
            account=account,
        )
        result = poller.result()  # Wait for completion

        print(f"  [OK] Resource created successfully")
        print(f"  Endpoint: {result.properties.endpoint}")
        print(f"  Provisioning state: {result.properties.provisioning_state}")

    except HttpResponseError as e:
        if "already exists" in str(e).lower() or e.status_code == 409:
            print(f"  [INFO] Resource already exists - skipping creation")
        else:
            print(f"  [ERROR] Creation failed: {e.message}")
            raise


# ---------------------------------------------------------------------------
# Create a single-service Azure OpenAI resource
# ---------------------------------------------------------------------------

def create_openai_resource(
    client: CognitiveServicesManagementClient,
    openai_account_name: str = None
) -> None:
    """
    Create a single-service Azure OpenAI resource.
    Kind='OpenAI' is specific to Azure OpenAI Service.
    """
    resource_name = openai_account_name or f"{ACCOUNT_NAME}-openai"
    print(f"\n--- Creating Azure OpenAI Resource ---")
    print(f"  Name: {resource_name}")
    print(f"  Kind: OpenAI (single-service)")

    account = Account(
        location=LOCATION,
        kind="OpenAI",
        sku=Sku(name="S0"),
        properties=AccountProperties(
            custom_sub_domain_name=resource_name.lower().replace("_", "-"),
        ),
    )

    try:
        poller = client.accounts.begin_create(
            resource_group_name=RESOURCE_GROUP,
            account_name=resource_name,
            account=account,
        )
        result = poller.result()
        print(f"  [OK] Azure OpenAI resource created")
        print(f"  Endpoint: {result.properties.endpoint}")

    except HttpResponseError as e:
        if "already exists" in str(e).lower() or e.status_code == 409:
            print(f"  [INFO] OpenAI resource already exists - skipping")
        else:
            print(f"  [ERROR] {e.message}")


# ---------------------------------------------------------------------------
# List all AI resources in the resource group
# ---------------------------------------------------------------------------

def list_ai_resources(client: CognitiveServicesManagementClient) -> None:
    """
    List all Cognitive Services / AI resources in the resource group.
    Useful for auditing what's deployed and finding endpoints.
    """
    print("\n--- Listing AI Resources in Resource Group ---")
    print(f"  Resource Group: {RESOURCE_GROUP}")

    try:
        accounts = client.accounts.list_by_resource_group(RESOURCE_GROUP)
        found = False
        for account in accounts:
            found = True
            print(f"\n  Resource: {account.name}")
            print(f"    Kind:     {account.kind}")
            print(f"    SKU:      {account.sku.name}")
            print(f"    Location: {account.location}")
            print(f"    State:    {account.properties.provisioning_state}")
            if account.properties.endpoint:
                print(f"    Endpoint: {account.properties.endpoint}")
            if account.properties.capabilities:
                caps = [c.name for c in account.properties.capabilities[:5]]
                print(f"    Capabilities: {caps}")

        if not found:
            print("  No AI resources found in this resource group")

    except HttpResponseError as e:
        print(f"  [ERROR] Failed to list resources: {e.message}")


# ---------------------------------------------------------------------------
# Get details of a specific resource
# ---------------------------------------------------------------------------

def get_resource_details(client: CognitiveServicesManagementClient) -> None:
    """
    Get detailed information about a specific AI resource.
    Includes endpoint, capabilities, and network rules.
    """
    print(f"\n--- Getting Resource Details: {ACCOUNT_NAME} ---")

    try:
        account = client.accounts.get(
            resource_group_name=RESOURCE_GROUP,
            account_name=ACCOUNT_NAME,
        )

        props = account.properties
        print(f"  Name:              {account.name}")
        print(f"  Kind:              {account.kind}")
        print(f"  SKU:               {account.sku.name}")
        print(f"  Location:          {account.location}")
        print(f"  Provisioning:      {props.provisioning_state}")
        print(f"  Endpoint:          {props.endpoint}")
        print(f"  Disable local auth: {props.disable_local_auth}")

        if props.network_acls:
            print(f"  Network ACLs:      {props.network_acls.default_action}")

        if props.capabilities:
            print(f"  Capabilities ({len(props.capabilities)} total):")
            for cap in props.capabilities[:8]:
                print(f"    - {cap.name}: {cap.value}")

    except HttpResponseError as e:
        if e.status_code == 404:
            print(f"  [INFO] Resource '{ACCOUNT_NAME}' not found")
            print("  Run create_multi_service_resource() first")
        else:
            print(f"  [ERROR] {e.message}")


# ---------------------------------------------------------------------------
# Check available resource types in a region
# ---------------------------------------------------------------------------

def list_available_kinds(client: CognitiveServicesManagementClient) -> None:
    """
    List available resource kinds and SKUs for a specific location.
    Use this to check what's available before creating resources.
    """
    print(f"\n--- Available AI Resource Kinds in {LOCATION} ---")

    try:
        resource_skus = client.resource_skus.list()
        seen_kinds = set()

        for sku in resource_skus:
            if sku.locations and LOCATION.lower() in [loc.lower() for loc in sku.locations]:
                kind_sku = f"{sku.kind}/{sku.name}"
                if kind_sku not in seen_kinds:
                    seen_kinds.add(kind_sku)
                    print(f"  Kind: {sku.kind:<30} SKU: {sku.name}")

    except HttpResponseError as e:
        print(f"  [ERROR] {e.message}")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Resource Management Demo")
    print("=" * 60)
    print(f"Subscription: {SUBSCRIPTION_ID[:8]}...")
    print(f"Resource Group: {RESOURCE_GROUP}")
    print(f"Location: {LOCATION}")

    client = create_management_client()

    # List existing resources first
    list_ai_resources(client)

    # Create a multi-service resource
    create_multi_service_resource(client)

    # Get details of the created resource
    get_resource_details(client)

    # List available kinds (informational)
    # list_available_kinds(client)  # Uncomment if needed (can be slow)

    print("\n" + "=" * 60)
    print("Resource Management Demo Complete")
    print("=" * 60)
