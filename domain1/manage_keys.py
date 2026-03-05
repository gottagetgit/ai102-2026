"""
manage_keys.py
==============
Demonstrates listing and regenerating API keys for an Azure AI Services resource
using the Azure Management SDK (azure-mgmt-cognitiveservices).

Exam Skill: "Manage and protect account keys" (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Listing current API keys for a Cognitive Services account
  - Regenerating key1 or key2 (key rotation pattern)
  - Implementing a safe key rotation strategy (rotate secondary → update apps → rotate primary)
  - Why key rotation matters for security compliance

Key rotation best practices demonstrated:
  1. Always maintain two keys so you can rotate without downtime
  2. Rotate key2 first, update all apps to use key2, then rotate key1
  3. Prefer Entra ID (keyless) auth to avoid key management entirely

Required packages:
  pip install azure-mgmt-cognitiveservices azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_SUBSCRIPTION_ID   - Your Azure subscription ID
  AZURE_RESOURCE_GROUP    - Resource group containing the AI resource
  AZURE_AI_ACCOUNT_NAME   - Name of the Cognitive Services account
"""

import os
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import RegenerateKeyParameters, KeyName
from azure.core.exceptions import AzureError

load_dotenv()

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP  = os.environ["AZURE_RESOURCE_GROUP"]
ACCOUNT_NAME    = os.environ["AZURE_AI_ACCOUNT_NAME"]


def get_client() -> CognitiveServicesManagementClient:
    """Create an authenticated management client using DefaultAzureCredential."""
    credential = DefaultAzureCredential()
    return CognitiveServicesManagementClient(credential, SUBSCRIPTION_ID)


def mask_key(key: str, visible_chars: int = 6) -> str:
    """
    Partially mask a key for safe display in logs.
    Never log full API keys - this helper shows just enough to identify which key is active.
    """
    if not key or len(key) <= visible_chars:
        return "***"
    return key[:visible_chars] + "*" * (len(key) - visible_chars)


def list_keys(client: CognitiveServicesManagementClient) -> dict:
    """
    List the current API keys for the Cognitive Services account.
    Returns a dict with 'key1' and 'key2'.

    The management plane returns plaintext keys - handle with care.
    In production, consider storing keys in Azure Key Vault rather than
    hardcoding them or putting them in .env files.
    """
    print(f"\n[LIST KEYS] Fetching keys for '{ACCOUNT_NAME}' ...")
    keys = client.accounts.list_keys(RESOURCE_GROUP, ACCOUNT_NAME)
    result = {"key1": keys.key1, "key2": keys.key2}
    print(f"  Key 1 : {mask_key(result['key1'])}")
    print(f"  Key 2 : {mask_key(result['key2'])}")
    return result


def regenerate_key(client: CognitiveServicesManagementClient, key_name: str) -> str:
    """
    Regenerate one of the API keys (key1 or key2).
    The old key becomes invalid immediately after regeneration.

    Args:
        key_name: "Key1" or "Key2"

    Returns:
        The new key value (as a string)
    """
    print(f"\n[REGENERATE] Regenerating {key_name} for '{ACCOUNT_NAME}' ...")

    # KeyName enum has values "Key1" and "Key2"
    params = RegenerateKeyParameters(key_name=key_name)
    keys = client.accounts.regenerate_key(RESOURCE_GROUP, ACCOUNT_NAME, parameters=params)

    new_key = keys.key1 if key_name == "Key1" else keys.key2
    print(f"  New {key_name}: {mask_key(new_key)}")
    print(f"  [WARNING] The old {key_name} is now INVALID. Update any apps using it immediately.")
    return new_key


def safe_key_rotation_demo(client: CognitiveServicesManagementClient) -> None:
    """
    Demonstrates a zero-downtime key rotation strategy.

    The pattern:
      Phase 1: Rotate key2 (secondary)
        - All apps still work using key1
        - Update apps to start using new key2

      Phase 2: Confirm apps use key2 (simulate by waiting)

      Phase 3: Rotate key1 (primary)
        - All apps now use key2 which is valid
        - Update apps to switch back to key1 if desired

    This ensures no downtime: at every point, at least one valid key exists.
    """
    print("\n" + "=" * 60)
    print("Zero-Downtime Key Rotation Demo")
    print("=" * 60)

    # Step 0: Show current state
    print("\n[PHASE 0] Current key state:")
    before = list_keys(client)

    # Step 1: Rotate secondary key (key2)
    print("\n[PHASE 1] Rotating Key2 (secondary) ...")
    print("  Apps continue running with Key1 while we rotate Key2.")
    new_key2 = regenerate_key(client, "Key2")

    print("\n  --> ACTION REQUIRED: Update your applications to use new Key2.")
    print("  --> In production, update Key Vault secret here.")
    # In real code: update_key_vault_secret("AI-Services-Key2", new_key2)

    # Simulate app update time
    print("\n  [Simulating app deployment with new key2 - 2 second pause] ...")
    time.sleep(2)

    # Step 2: Verify Key1 still valid (list keys to show current state)
    print("\n[PHASE 2] Verifying current key state:")
    list_keys(client)
    print("  Key1 is still valid - apps currently using Key1 are unaffected.")

    # Step 3: Now safe to rotate Key1 (primary)
    print("\n[PHASE 3] Rotating Key1 (primary) ...")
    print("  Apps have been switched to Key2, so Key1 rotation is safe.")
    new_key1 = regenerate_key(client, "Key1")

    print("\n  --> ACTION REQUIRED: Optionally switch apps back to Key1.")
    print("  --> Update Key Vault secret with new Key1 value.")

    # Step 4: Final state
    print("\n[PHASE 4] Final key state after rotation:")
    list_keys(client)

    print("\n[ROTATION COMPLETE]")
    print("  Both keys have been rotated. Apps should now use the new keys.")
    print("  Old key1 and old key2 are no longer valid.")


def show_key_vault_pattern() -> None:
    """
    Prints guidance on the recommended Key Vault integration pattern.
    This is the production-grade approach: never put keys in code or .env files.
    """
    print("\n" + "=" * 60)
    print("Recommended Pattern: Store Keys in Azure Key Vault")
    print("=" * 60)
    print("""
  Instead of storing AI Services keys in .env files or config,
  use Azure Key Vault with automatic rotation:

  1. Store the key in Key Vault as a secret:
       az keyvault secret set \\
         --vault-name <vault-name> \\
         --name "AiServicesKey1" \\
         --value "<key>"

  2. Grant your app's managed identity access to Key Vault:
       az keyvault set-policy \\
         --name <vault-name> \\
         --object-id <app-managed-identity-object-id> \\
         --secret-permissions get

  3. In Python, retrieve the key at runtime:
       from azure.keyvault.secrets import SecretClient
       from azure.identity import DefaultAzureCredential

       client = SecretClient(
           vault_url="https://<vault-name>.vault.azure.net",
           credential=DefaultAzureCredential()
       )
       secret = client.get_secret("AiServicesKey1")
       api_key = secret.value

  4. Even better: use Entra ID (keyless) authentication entirely,
     which eliminates key management:
       from azure.identity import DefaultAzureCredential
       from azure.ai.textanalytics import TextAnalyticsClient

       credential = DefaultAzureCredential()
       client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

  See authenticate_entra.py for the keyless authentication pattern.
""")


def main():
    print("=" * 60)
    print("Azure AI Services Key Management Demo")
    print("=" * 60)

    try:
        client = get_client()

        # 1. List current keys
        list_keys(client)

        # 2. Show the safe rotation strategy (will actually rotate keys)
        print("\n[NOTE] The rotation demo will regenerate both keys.")
        print("       Comment out 'safe_key_rotation_demo()' if you don't want that.")
        safe_key_rotation_demo(client)

        # 3. Show the Key Vault best-practice pattern
        show_key_vault_pattern()

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except AzureError as e:
        print(f"\n[ERROR] Azure Management API error: {e}")
        print("Ensure your identity has 'Cognitive Services Contributor' role on the resource.")


if __name__ == "__main__":
    main()
