"""
create_ai_resource.py
=====================
Demonstrates creating and managing an Azure AI Services (Cognitive Services) resource
using the Azure Management SDK.

Exam Skill: "Create an Azure AI resource" (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Authenticating with Azure using DefaultAzureCredential
  - Creating a multi-service Azure AI Services resource in a resource group
  - Listing all Cognitive Services accounts in a subscription
  - Retrieving properties of a specific resource
  - Deleting a resource (commented out for safety)

Required packages:
  pip install azure-mgmt-cognitiveservices azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_SUBSCRIPTION_ID   - Your Azure subscription ID
  AZURE_RESOURCE_GROUP    - Target resource group name
  AZURE_LOCATION          - Azure region e.g. "eastus"
  AZURE_AI_ACCOUNT_NAME   - Name for the new AI Services account
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
from azure.core.exceptions import AzureError, ResourceExistsError

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
SUBSCRIPTION_ID  = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP   = os.environ["AZURE_RESOURCE_GROUP"]
LOCATION         = os.environ.get("AZURE_LOCATION", "eastus")
ACCOUNT_NAME     = os.environ["AZURE_AI_ACCOUNT_NAME"]

# SKU options for Azure AI Services: F0 (free), S0 (standard)
SKU_NAME = "S0"

# The kind "CognitiveServices" creates a multi-service AI resource.
# Other kind values: "OpenAI", "TextAnalytics", "ComputerVision", etc.
ACCOUNT_KIND = "CognitiveServices"


def get_client() -> CognitiveServicesManagementClient:
    """Create and return an authenticated CognitiveServicesManagementClient."""
    credential = DefaultAzureCredential()
    return CognitiveServicesManagementClient(credential, SUBSCRIPTION_ID)


def create_ai_services_resource(client: CognitiveServicesManagementClient) -> Account:
    """
    Create a new Azure AI Services (multi-service) resource.

    The 'CognitiveServices' kind bundles Language, Vision, Speech, and other
    services under a single endpoint and key - ideal for solutions that use
    multiple AI capabilities.
    """
    print(f"\n[CREATE] Creating Azure AI Services resource '{ACCOUNT_NAME}' ...")
    print(f"  Subscription : {SUBSCRIPTION_ID}")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Location     : {LOCATION}")
    print(f"  SKU          : {SKU_NAME}")
    print(f"  Kind         : {ACCOUNT_KIND}")

    account_params = Account(
        kind=ACCOUNT_KIND,
        sku=Sku(name=SKU_NAME),
        location=LOCATION,
        properties=AccountProperties(
            public_network_access="Enabled",
            # Disable local auth (API key) and require Entra ID only - best practice
            # disable_local_auth=True,  # Uncomment to enforce keyless auth
        ),
        tags={
            "purpose": "ai102-exam-demo",
            "managed_by": "python-sdk",
        },
    )

    try:
        account = client.accounts.create(
            resource_group_name=RESOURCE_GROUP,
            account_name=ACCOUNT_NAME,
            account=account_params,
        )
        print(f"[CREATE] Resource created successfully.")
        print(f"  ID       : {account.id}")
        print(f"  Endpoint : {account.properties.endpoint}")
        print(f"  State    : {account.properties.provisioning_state}")
        return account
    except ResourceExistsError:
        print(f"[CREATE] Resource '{ACCOUNT_NAME}' already exists - retrieving it.")
        return client.accounts.get(RESOURCE_GROUP, ACCOUNT_NAME)


def list_ai_resources(client: CognitiveServicesManagementClient) -> None:
    """
    List all Cognitive Services / Azure AI resources in the subscription.
    Useful for auditing what AI resources exist across resource groups.
    """
    print("\n[LIST] All Azure AI Services accounts in subscription:")
    accounts = list(client.accounts.list())
    if not accounts:
        print("  No accounts found.")
        return
    for acc in accounts:
        print(f"  - {acc.name:30s} | Kind: {acc.kind:20s} | Location: {acc.location:15s} | SKU: {acc.sku.name}")


def get_resource_properties(client: CognitiveServicesManagementClient, name: str) -> None:
    """
    Retrieve and display detailed properties of a specific AI resource.
    Shows endpoint, provisioning state, capabilities, and network rules.
    """
    print(f"\n[GET] Properties of '{name}':")
    try:
        account = client.accounts.get(RESOURCE_GROUP, name)
        props = account.properties

        print(f"  Provisioning State : {props.provisioning_state}")
        print(f"  Endpoint           : {props.endpoint}")
        print(f"  Custom Domain      : {props.custom_sub_domain_name}")
        print(f"  Public Network     : {props.public_network_access}")
        print(f"  Disable Local Auth : {props.disable_local_auth}")
        print(f"  SKU                : {account.sku.name}")
        print(f"  Kind               : {account.kind}")
        print(f"  Location           : {account.location}")
        print(f"  Tags               : {account.tags}")

        # Show available capabilities (e.g., which sub-services are supported)
        if props.capabilities:
            print("  Capabilities:")
            for cap in props.capabilities:
                print(f"    {cap.name}: {cap.value}")
    except AzureError as e:
        print(f"  [ERROR] Could not retrieve resource: {e}")


def list_available_skus(client: CognitiveServicesManagementClient) -> None:
    """
    List available SKUs for Cognitive Services in the target location.
    Useful when planning capacity and choosing the right tier.
    """
    print(f"\n[SKUS] Available SKUs in '{LOCATION}' for kind '{ACCOUNT_KIND}':")
    skus = list(client.resource_skus.list())
    shown = 0
    for sku in skus:
        if sku.kind == ACCOUNT_KIND:
            locs = sku.locations or []
            if LOCATION.lower() in [loc.lower() for loc in locs]:
                print(f"  SKU: {sku.name:10s} | Restrictions: {len(sku.restrictions or [])} restriction(s)")
                shown += 1
    if shown == 0:
        print("  No matching SKUs found (try adjusting LOCATION or ACCOUNT_KIND).")


# ---------------------------------------------------------------------------
# Uncomment to delete the resource (destructive - use with caution)
# ---------------------------------------------------------------------------
# def delete_ai_resource(client: CognitiveServicesManagementClient, name: str) -> None:
#     """Delete an Azure AI Services resource."""
#     print(f"\n[DELETE] Deleting '{name}' ...")
#     client.accounts.delete(RESOURCE_GROUP, name)
#     print("  Deleted.")


def main():
    print("=" * 60)
    print("Azure AI Services Resource Management Demo")
    print("=" * 60)

    try:
        client = get_client()

        # 1. Create the resource
        create_ai_services_resource(client)

        # 2. List all AI resources in the subscription
        list_ai_resources(client)

        # 3. Get properties of the resource we just created
        get_resource_properties(client, ACCOUNT_NAME)

        # 4. List available SKUs
        list_available_skus(client)

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
        print("Make sure your .env file contains all required variables.")
    except AzureError as e:
        print(f"\n[ERROR] Azure error: {e}")
        print("Check that your credentials have Contributor or Cognitive Services Contributor role.")


if __name__ == "__main__":
    main()
