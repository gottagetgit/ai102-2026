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
    Authenticate using a service principal (client ID + secret).

    Best for: CI/CD pipelines, automation scripts, cross-tenant scenarios.
    Create a service principal: az ad sp create-for-rbac --name ai102-demo
    Assign role: az role assignment create --role "Cognitive Services User" ...
    """
    print("\n--- Method 3: Service Principal Authentication ---")

    # These would typically come from environment variables or a secrets manager
    tenant_id = os.environ.get("AZURE_TENANT_ID", "")
    client_id = os.environ.get("AZURE_CLIENT_ID", "")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")

    if not all([tenant_id, client_id, client_secret]):
        print("  [SKIP] Service principal env vars not set (AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)")
        print("  This is expected - set these to test service principal auth.")
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

        result = client.detect_language(["Hola, como estas?"])
        lang = result[0].primary_language
        print(f"  Language='{lang.name}' (confidence={lang.confidence_score:.2f})")
        print("  [OK] Service principal authentication successful.")

    except ClientAuthenticationError as e:
        print(f"  [ERROR] Service principal auth failed: {e.message}")


# ---------------------------------------------------------------------------
# Method 4: Managed Identity - for Azure-hosted workloads
# ---------------------------------------------------------------------------

def demo_managed_identity_auth() -> None:
    """
    Demonstrate managed identity authentication (system-assigned or user-assigned).

    This only works when running on Azure (VMs, App Service, Functions, AKS, etc.).
    The identity must have the appropriate RBAC role on the AI resource.

    For user-assigned managed identity, specify the client_id.
    """
    print("\n--- Method 4: Managed Identity Authentication ---")
    print("  Note: This only works when running on an Azure-hosted service.")
    print("  Running locally will show a timeout/failure - this is expected.")

    try:
        # System-assigned managed identity (no parameters needed)
        # For user-assigned: ManagedIdentityCredential(client_id="<user-assigned-client-id>")
        credential = ManagedIdentityCredential()

        client = TextAnalyticsClient(
            endpoint=AI_ENDPOINT,
            credential=credential,
        )

        result = client.detect_language(["Ciao, come stai?"])
        lang = result[0].primary_language
        print(f"  Language='{lang.name}' (confidence={lang.confidence_score:.2f})")
        print("  [OK] Managed identity authentication successful.")

    except ClientAuthenticationError as e:
        print(f"  [INFO] Managed identity not available (expected when running locally).")
        print(f"  Details: {e.message[:100]}...")
    except Exception as e:
        print(f"  [INFO] Managed identity error (expected locally): {type(e).__name__}")


# ---------------------------------------------------------------------------
# Method 5: Chained Credentials - custom fallback chain
# ---------------------------------------------------------------------------

def demo_chained_credentials() -> None:
    """
    Build a custom credential chain using ChainedTokenCredential.

    Use this when you want specific fallback behavior different from
    DefaultAzureCredential's built-in chain.
    """
    print("\n--- Method 5: Chained Credentials ---")
    print("  Chain: ManagedIdentity → AzureCLI (custom fallback)")

    try:
        credential = ChainedTokenCredential(
            ManagedIdentityCredential(),
            AzureCliCredential(),
        )

        client = TextAnalyticsClient(
            endpoint=AI_ENDPOINT,
            credential=credential,
        )

        result = client.detect_language(["Guten Tag, wie geht es Ihnen?"])
        lang = result[0].primary_language
        print(f"  Language='{lang.name}' (confidence={lang.confidence_score:.2f})")
        print("  [OK] Chained credential authentication successful.")

    except ClientAuthenticationError as e:
        print(f"  [INFO] Chained credential failed: no credential in chain succeeded.")
        print("  Run 'az login' to enable the AzureCliCredential fallback.")


# ---------------------------------------------------------------------------
# Summary: When to use each method
# ---------------------------------------------------------------------------

def print_auth_summary() -> None:
    summary = """
=== Azure AI Authentication Methods Summary ===

Method              | When to Use
--------------------|--------------------------------------------------
API Key             | Simple dev/test; when Entra ID is unavailable
DefaultAzureCredential | Recommended for ALL environments; adapts automatically
Service Principal   | CI/CD pipelines; automation; cross-tenant
Managed Identity    | Production Azure workloads (VMs, App Service, etc.)
Chained Credential  | Custom fallback order; specific environment needs

Best Practice:
  - Local dev:    az login + DefaultAzureCredential
  - Azure hosted: System-assigned managed identity + DefaultAzureCredential
  - CI/CD:        Service principal via AZURE_CLIENT_ID/SECRET/TENANT_ID env vars

RBAC Roles needed:
  - Cognitive Services User         - read/call AI services
  - Cognitive Services Contributor  - manage resources
  - Cognitive Services OpenAI User  - call Azure OpenAI
  - Cognitive Services OpenAI Contributor - fine-tune models
"""
    print(summary)


if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Services Authentication Methods Demo")
    print("=" * 60)

    demo_api_key_auth()
    demo_openai_api_key_auth()
    demo_default_azure_credential()
    demo_openai_entra_auth()
    demo_service_principal_auth()
    demo_managed_identity_auth()
    demo_chained_credentials()
    print_auth_summary()
