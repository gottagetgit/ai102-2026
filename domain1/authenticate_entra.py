"""
authenticate_entra.py
=====================
Demonstrates two authentication patterns for Azure AI Services:
  1. API Key authentication (simple but less secure)
  2. Microsoft Entra ID (keyless) authentication using DefaultAzureCredential

Exam Skill: "Manage authentication for a Microsoft Foundry Service resource"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Authenticating with an API key (AzureKeyCredential)
  - Authenticating with Entra ID via DefaultAzureCredential (recommended)
  - DefaultAzureCredential credential chain (how it picks the right credential)
  - Setting up a managed identity for keyless auth in production
  - Using token-based auth with the Azure OpenAI SDK
  - Why keyless auth is preferred (no secret management, automatic rotation)

DefaultAzureCredential tries credentials in this order:
  1. EnvironmentCredential      (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
  2. WorkloadIdentityCredential (Kubernetes workload identity)
  3. ManagedIdentityCredential  (Azure VM / App Service managed identity)
  4. SharedTokenCacheCredential (cached token from Visual Studio / VS Code)
  5. VisualStudioCredential     (Visual Studio sign-in)
  6. AzureCliCredential         (az login)
  7. AzurePowerShellCredential  (Connect-AzAccount)
  8. AzureDeveloperCliCredential (azd auth login)
  9. InteractiveBrowserCredential (browser popup - disabled by default)

Required packages:
  pip install azure-identity azure-ai-textanalytics openai python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT  - e.g. https://<name>.cognitiveservices.azure.com/
  AZURE_AI_SERVICES_KEY       - API key (for key-based auth demo only)
  AZURE_OPENAI_ENDPOINT       - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY            - API key (for key-based auth demo only)
  AZURE_OPENAI_DEPLOYMENT     - Deployment name e.g. "gpt-4o"
"""

import os
from dotenv import load_dotenv

# Azure Identity - core of keyless auth
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    ClientSecretCredential,
    AzureCliCredential,
    ChainedTokenCredential,
    get_bearer_token_provider,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

# AI SDK clients we will authenticate
from azure.ai.textanalytics import TextAnalyticsClient

# OpenAI SDK with Azure
import openai

load_dotenv()

AI_ENDPOINT     = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
AI_KEY          = os.environ["AZURE_AI_SERVICES_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY      = os.environ["AZURE_OPENAI_KEY"]
OPENAI_DEPLOY   = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Cognitive Services scope required for token-based auth
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


# ---------------------------------------------------------------------------
# Method 1: API Key Authentication
# ---------------------------------------------------------------------------

def demo_api_key_auth() -> None:
    """
    Authenticate using an API key (AzureKeyCredential).

    Pros:
      - Simple, works immediately
      - No Azure AD setup needed

    Cons:
      - Keys must be stored securely (Key Vault or env vars)
      - Keys can be leaked if not handled carefully
      - Requires manual rotation
      - Cannot be scoped to specific operations

    When to use: Development/testing, or when Entra ID is not available.
    """
    print("\n--- Method 1: API Key Authentication ---")
    print("Using AzureKeyCredential with API key from environment variable")

    client = TextAnalyticsClient(
        endpoint=AI_ENDPOINT,
        credential=AzureKeyCredential(AI_KEY),
    )

    try:
        result = client.detect_language(["Hello, how are you?"])
        lang = result[0].primary_language
        print(f"  Text Analytics response: Language='{lang.name}' (confidence={lang.confidence_score:.2f})")
        print("  [OK] API key authentication successful.")
    except HttpResponseError as e:
        print(f"  [ERROR] API call failed: {e.message}")


def demo_openai_api_key_auth() -> None:
    """
    Authenticate to Azure OpenAI using an API key.
    The openai package uses api_key + azure_endpoint configuration.
    """
    print("\n--- Method 1b: Azure OpenAI API Key Authentication ---")

    az_client = openai.AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-02-01",
    )

    try:
        response = az_client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[{"role": "user", "content": "Say 'API key auth works' in 5 words."}],
            max_tokens=20,
        )
        print(f"  Response: {response.choices[0].message.content}")
        print("  [OK] Azure OpenAI API key authentication successful.")
    except openai.AuthenticationError as e:
        print(f"  [ERROR] Authentication failed: {e}")
    except openai.APIError as e:
        print(f"  [ERROR] API error: {e}")


# ---------------------------------------------------------------------------
# Method 2: Microsoft Entra ID (keyless) - DefaultAzureCredential
# ---------------------------------------------------------------------------

def demo_default_azure_credential() -> None:
    """
    Authenticate using DefaultAzureCredential - the recommended approach.

    This tries multiple credential sources in order and uses the first one
    that succeeds. In local development, it typically uses AzureCliCredential
    (from 'az login'). In production on Azure, it uses ManagedIdentityCredential.

    Required RBAC role: "Cognitive Services User" on the AI Services resource.
    """
    print("\n--- Method 2: Entra ID - DefaultAzureCredential ---")
    print("Trying credential chain: EnvironmentCredential → ManagedIdentity → AzureCLI → ...")

    try:
        credential = DefaultAzureCredential()

        client = TextAnalyticsClient(
            endpoint=AI_ENDPOINT,
            credential=credential,
        )

        result = client.detect_language(["Bonjour, comment allez-vous?"])
        lang = result[0].primary_language
        print(f"  Text Analytics response: Language='{lang.name}' (confidence={lang.confidence_score:.2f})")
        print("  [OK] DefaultAzureCredential authentication successful.")
        print("  Active credential: check Azure CLI 'az account show' for current identity")

    except ClientAuthenticationError as e:
        print(f"  [INFO] DefaultAzureCredential failed - no valid credential found in chain.")
        print(f"  Details: {e.message}")
        print("  To fix: run 'az login' or set AZURE_CLIENT_ID/SECRET/TENANT_ID env vars.")


def demo_openai_entra_auth() -> None:
    """
    Authenticate to Azure OpenAI using Entra ID (keyless).

    The openai package supports token-based auth via azure_ad_token_provider.
    Use get_bearer_token_provider() from azure-identity to create the provider.

    Required RBAC role: "Cognitive Services OpenAI User" on the Azure OpenAI resource.
    """
    print("\n--- Method 2b: Azure OpenAI Entra ID Authentication ---")

    try:
        credential = DefaultAzureCredential()

        # Create a token provider function that the openai SDK calls to get fresh tokens
        token_provider = get_bearer_token_provider(
            credential,
            COGNITIVE_SERVICES_SCOPE,
        )

        az_client = openai.AzureOpenAI(
            azure_ad_token_provider=token_provider,  # No api_key needed
            azure_endpoint=OPENAI_ENDPOINT,
            api_version="2024-02-01",
        )

        response = az_client.chat.completions.create(
            model=OPENAI_DEPLOY,
            messages=[{"role": "user", "content": "Say 'Entra ID auth works' in 5 words."}],
            max_tokens=20,
        )
        print(f"  Response: {response.choices[0].message.content}")
        print("  [OK] Azure OpenAI Entra ID (keyless) authentication successful.")

    except ClientAuthenticationError as e:
        print(f"  [INFO] Entra auth failed: {e.message}")
        print("  Ensure you have 'Cognitive Services OpenAI User' role and ran 'az login'.")
    except openai.AuthenticationError as e:
        print(f"  [ERROR] OpenAI auth failed: {e}")


# ---------------------------------------------------------------------------
# Method 3: Service Principal (client secret) - for automated pipelines
# ---------------------------------------------------------------------------

def demo_service_principal_auth() -> None:
    """
    Authenticate using a service principal with client secret.
    Used for non-interactive scenarios like CI/CD pipelines.

    Required env vars (not in .env - set in pipeline secrets):
      AZURE_TENANT_ID      - Azure AD tenant ID
      AZURE_CLIENT_ID      - Service principal app (client) ID
      AZURE_CLIENT_SECRET  - Service principal secret

    Required RBAC role: "Cognitive Services User" on the AI Services resource.
    """
    print("\n--- Method 3: Service Principal (Client Secret) ---")

    tenant_id     = os.environ.get("AZURE_TENANT_ID")
    client_id     = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")

    if not all([tenant_id, client_id, client_secret]):
        print("  [SKIP] AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET not set.")
        print("  This method requires a service principal with client secret.")
        print(
            "  To create one:\n"
            "    az ad sp create-for-rbac --name ai102-demo \\\n"
            "      --role 'Cognitive Services User' \\\n"
            "      --scopes /subscriptions/<sub>/resourceGroups/<rg>/providers/"
            "Microsoft.CognitiveServices/accounts/<name>"
        )
        return

    try:
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        client = TextAnalyticsClient(
            endpoint=AI_ENDPOINT,
            credential=credential,
        )

        result = client.detect_language(["Hola, ¿cómo estás?"])
        lang = result[0].primary_language
        print(f"  Language: {lang.name} (confidence: {lang.confidence_score:.2f})")
        print("  [OK] Service principal authentication successful.")

    except ClientAuthenticationError as e:
        print(f"  [ERROR] Service principal auth failed: {e.message}")


def demo_managed_identity_guidance() -> None:
    """
    Prints guidance on setting up managed identity for production deployments.
    This is the gold standard for keyless auth in Azure-hosted applications.
    """
    print("\n--- Method 4: Managed Identity (Production Recommended) ---")
    print("""
  System-assigned managed identity (for Azure VM, App Service, ACI, AKS):
  
    Step 1: Enable managed identity on your Azure resource:
      az webapp identity assign --name <app> --resource-group <rg>
      # or for VM:
      az vm identity assign --name <vm> --resource-group <rg>
    
    Step 2: Grant the managed identity access to AI Services:
      az role assignment create \\
        --assignee <managed-identity-object-id> \\
        --role "Cognitive Services User" \\
        --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/
                Microsoft.CognitiveServices/accounts/<name>
    
    Step 3: In code - ManagedIdentityCredential is auto-detected by DefaultAzureCredential:
      from azure.identity import DefaultAzureCredential
      credential = DefaultAzureCredential()
      # No keys, no secrets, no rotation needed!
  
  User-assigned managed identity (for shared identity across multiple resources):
      credential = ManagedIdentityCredential(client_id="<user-assigned-mi-client-id>")
  
  Benefits:
    - No secrets to store, rotate, or leak
    - Azure AD manages the credential lifecycle
    - Auditable via Azure AD sign-in logs
    - Works with Conditional Access policies
""")


def main():
    print("=" * 60)
    print("Azure AI Services Authentication Patterns Demo")
    print("=" * 60)
    print(
        "\nThis demo shows both API key (simple) and Entra ID (recommended)\n"
        "authentication approaches for Azure AI Services.\n"
    )

    # Method 1: API Key
    demo_api_key_auth()
    demo_openai_api_key_auth()

    # Method 2: Entra ID (DefaultAzureCredential)
    demo_default_azure_credential()
    demo_openai_entra_auth()

    # Method 3: Service Principal
    demo_service_principal_auth()

    # Method 4: Managed Identity guidance
    demo_managed_identity_guidance()

    print("\n" + "=" * 60)
    print("Authentication Pattern Summary")
    print("=" * 60)
    print("""
  | Scenario                   | Recommended Method           |
  |----------------------------|------------------------------|
  | Local development          | AzureCliCredential (az login)|
  | CI/CD pipeline             | ClientSecretCredential       |
  | Azure VM / App Service     | ManagedIdentityCredential    |
  | AKS                        | WorkloadIdentityCredential   |
  | Any Azure-hosted resource  | DefaultAzureCredential       |
  | Quick test/prototype       | AzureKeyCredential (key)     |
""")


if __name__ == "__main__":
    main()
