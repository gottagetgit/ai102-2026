"""
manage_keys.py
==============
Demonstrates listing and rotating API keys for Azure AI Services
using the Azure Management SDK.

Exam Skill: "Manage and protect account keys"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Listing both API keys for a resource
  - Regenerating (rotating) key1 or key2
  - Zero-downtime key rotation pattern (rotate one at a time)
  - Disabling local (key-based) auth to enforce Entra ID only
  - Azure Key Vault integration guidance (where to store keys)
  - Best practices for key management

Azure AI Services has TWO keys (key1 and key2):
  - Always rotate one at a time to avoid service disruption
  - Update your app to use the new key before rotating the other
  - Best practice: store keys in Azure Key Vault, not in .env files

Required packages:
  pip install azure-identity azure-mgmt-cognitiveservices python-dotenv

Required environment variables (in .env):
  AZURE_SUBSCRIPTION_ID  - your Azure subscription ID
  AZURE_RESOURCE_GROUP   - resource group name
  AZURE_AI_ACCOUNT_NAME  - name of the AI Services account
"""

import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import RegenerateKeyParameters
from azure.core.exceptions import HttpResponseError

load_dotenv()

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP  = os.environ["AZURE_RESOURCE_GROUP"]
ACCOUNT_NAME    = os.environ["AZURE_AI_ACCOUNT_NAME"]


def create_management_client() -> CognitiveServicesManagementClient:
    """Create the management client using DefaultAzureCredential."""
    return CognitiveServicesManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
    )


# ---------------------------------------------------------------------------
# List API keys
# ---------------------------------------------------------------------------

def list_keys(client: CognitiveServicesManagementClient) -> dict:
    """
    List both API keys for the AI Services resource.
    Both keys are valid simultaneously - this enables zero-downtime rotation.
    """
    print("\n--- Listing API Keys ---")

    try:
        keys = client.accounts.list_keys(
            resource_group_name=RESOURCE_GROUP,
            account_name=ACCOUNT_NAME,
        )

        # Show masked keys for security
        key1_masked = f"{keys.key1[:8]}...{keys.key1[-4:]}" if keys.key1 else "None"
        key2_masked = f"{keys.key2[:8]}...{keys.key2[-4:]}" if keys.key2 else "None"

        print(f"  Key1: {key1_masked}")
        print(f"  Key2: {key2_masked}")
        print(f"  Both keys are valid simultaneously")

        return {"key1": keys.key1, "key2": keys.key2}

    except HttpResponseError as e:
        print(f"  [ERROR] Failed to list keys: {e.message}")
        return {}


# ---------------------------------------------------------------------------
# Rotate a key
# ---------------------------------------------------------------------------

def rotate_key(
    client: CognitiveServicesManagementClient,
    key_name: str = "Key1"
) -> str:
    """
    Regenerate (rotate) one of the API keys.

    Best practice: rotate key2 first, update app, then rotate key1.
    This ensures no downtime during rotation.

    Args:
        key_name: 'Key1' or 'Key2'
    Returns:
        The new key value
    """
    print(f"\n--- Rotating {key_name} ---")
    print(f"  Resource: {ACCOUNT_NAME}")

    try:
        result = client.accounts.regenerate_key(
            resource_group_name=RESOURCE_GROUP,
            account_name=ACCOUNT_NAME,
            parameters=RegenerateKeyParameters(key_name=key_name),
        )

        new_key = result.key1 if key_name == "Key1" else result.key2
        new_key_masked = f"{new_key[:8]}...{new_key[-4:]}" if new_key else "None"

        print(f"  [OK] {key_name} regenerated successfully")
        print(f"  New {key_name}: {new_key_masked}")
        print(f"  Update your .env / Key Vault with the new key!")

        return new_key

    except HttpResponseError as e:
        print(f"  [ERROR] Key rotation failed: {e.message}")
        return ""


# ---------------------------------------------------------------------------
# Zero-downtime key rotation pattern
# ---------------------------------------------------------------------------

def zero_downtime_rotation_demo(client: CognitiveServicesManagementClient) -> None:
    """
    Demonstrate the recommended zero-downtime key rotation pattern.

    Steps:
      1. Your app currently uses Key1
      2. Rotate Key2 (app still works on Key1)
      3. Update your app to use Key2
      4. Rotate Key1 (app still works on Key2)
      5. Optionally: update app back to Key1

    This pattern ensures no downtime during key rotation.
    """
    print("\n" + "=" * 60)
    print("ZERO-DOWNTIME KEY ROTATION PATTERN")
    print("=" * 60)

    print("""
Scenario: Your app is using Key1. You need to rotate both keys.

Step 1: App is running on Key1 (current state)
Step 2: Rotate Key2 (safe - app still uses Key1)
Step 3: Update your app/config to use new Key2
Step 4: Rotate Key1 (safe - app now uses Key2)
Step 5: (Optional) Update app back to new Key1

This ensures continuous availability throughout the rotation.
""")

    # Show current keys
    print("  [Step 1] Current keys:")
    keys_before = list_keys(client)

    if not keys_before:
        print("  Cannot proceed - failed to retrieve keys")
        return

    # Step 2: Rotate Key2
    print("\n  [Step 2] Rotating Key2 (app still on Key1)...")
    new_key2 = rotate_key(client, "Key2")

    # Step 3: Update app config
    print("\n  [Step 3] Update your app to use the new Key2")
    print("  In practice: update .env, Key Vault secret, or app config")
    print(f"  New Key2 to store: {new_key2[:8]}...")

    # Step 4: Rotate Key1
    print("\n  [Step 4] Rotating Key1 (app now on Key2)...")
    new_key1 = rotate_key(client, "Key1")

    print("\n  [OK] Zero-downtime rotation complete!")
    print(f"  Update your primary config to use new Key1: {new_key1[:8]}...")


# ---------------------------------------------------------------------------
# Disable local (key-based) authentication
# ---------------------------------------------------------------------------

def show_disable_local_auth_guidance() -> None:
    """
    Show how to disable key-based authentication (enforce Entra ID only).
    This is a security best practice for production environments.
    """
    print("\n" + "=" * 60)
    print("DISABLING LOCAL (KEY-BASED) AUTHENTICATION")
    print("=" * 60)

    print("""
To enforce Entra ID-only authentication (disable API keys):

  1. Via Azure Portal:
     - Go to your AI resource > Settings > Keys and Endpoint
     - Toggle 'Local authentication' to OFF

  2. Via Management SDK:
     from azure.mgmt.cognitiveservices.models import AccountProperties
     
     # Update the resource
     account = client.accounts.get(resource_group, account_name)
     account.properties.disable_local_auth = True
     client.accounts.begin_create(resource_group, account_name, account).result()

  3. Via Azure CLI:
     az cognitiveservices account update \\
       --name <account> --resource-group <rg> \\
       --custom-domain <domain> \\
       --api-properties disableLocalAuth=true

  4. Via Azure Policy:
     - Assign policy: 'Cognitive Services accounts should disable key access'
     - This enforces the setting across your subscription

When disabled:
  - API keys no longer work (401 Unauthorized)
  - Must use Entra ID (managed identity or service principal)
  - Assign RBAC role 'Cognitive Services User' to your identity
""")


# ---------------------------------------------------------------------------
# Key Vault integration guidance
# ---------------------------------------------------------------------------

def show_keyvault_guidance() -> None:
    """
    Show guidance for storing AI keys in Azure Key Vault.
    """
    print("\n" + "=" * 60)
    print("AZURE KEY VAULT INTEGRATION")
    print("=" * 60)

    print("""
Best Practice: Store API keys in Azure Key Vault, not in .env files.

  1. Store the key in Key Vault:
     az keyvault secret set \\
       --vault-name <vault> \\
       --name 'azure-ai-key1' \\
       --value '<your-key>'

  2. Retrieve in your app:
     from azure.keyvault.secrets import SecretClient
     from azure.identity import DefaultAzureCredential
     
     vault_url = 'https://<vault>.vault.azure.net'
     client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
     secret = client.get_secret('azure-ai-key1')
     api_key = secret.value

  3. For key rotation with Key Vault:
     - Use Key Vault references in App Service / Azure Functions
     - App automatically picks up new key after rotation
     - No redeployment needed

  4. Access policy needed:
     - Grant your app's managed identity 'Get' permission on secrets
     az keyvault set-policy \\
       --name <vault> \\
       --object-id <managed-identity-object-id> \\
       --secret-permissions get list
""")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Key Management Demo")
    print("=" * 60)
    print(f"Resource: {ACCOUNT_NAME}")
    print(f"Resource Group: {RESOURCE_GROUP}")

    client = create_management_client()

    # List current keys
    list_keys(client)

    # Demonstrate zero-downtime rotation
    # Uncomment to actually rotate keys (this will invalidate current keys!)
    # zero_downtime_rotation_demo(client)

    # Show security guidance
    show_disable_local_auth_guidance()
    show_keyvault_guidance()

    print("\n" + "=" * 60)
    print("Key Management Demo Complete")
    print("=" * 60)
