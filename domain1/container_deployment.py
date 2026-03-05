"""
container_deployment.py
=======================
Demonstrates pulling and running Azure AI Services containers locally
and calling the local container endpoint.

Exam Skill: "Plan and implement a container deployment"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Which Azure AI Services support containerization
  - Required parameters for ALL containers (Eula, Billing, ApiKey)
  - How to pull and run containers with docker run commands
  - Calling the local container endpoint (same API as cloud)
  - Health check endpoint for container readiness
  - Container configuration options (CPU, memory, port mapping)
  - Limitations: containers still phone home to Azure for billing

Supported Azure AI containers (subset):
  - azure-cognitive-services/textanalytics/language  (Language detection, sentiment)
  - azure-cognitive-services/textanalytics/keyphrase (Key phrase extraction)
  - azure-cognitive-services/textanalytics/ner       (Named entity recognition)
  - azure-cognitive-services/vision/read             (OCR / Read API)
  - azure-cognitive-services/speechservices/speech-to-text
  - azure-cognitive-services/speechservices/text-to-speech
  - azure-cognitive-services/form-recognizer/layout  (Document Intelligence)
  - azure-cognitive-services/translator              (Translator)

Required packages:
  pip install requests python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT - your Azure AI Services endpoint (for billing)
  AZURE_AI_SERVICES_KEY      - your Azure AI Services API key
"""

import os
import time
import json
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
AZURE_API_KEY  = os.environ["AZURE_AI_SERVICES_KEY"]

# Local container configuration
LOCAL_PORT        = 5000
LOCAL_CONTAINER_URL = f"http://localhost:{LOCAL_PORT}"
CONTAINER_IMAGE   = "mcr.microsoft.com/azure-cognitive-services/textanalytics/language:latest"
CONTAINER_NAME    = "ai102-language-demo"


# ---------------------------------------------------------------------------
# Section 1: Container reference documentation
# ---------------------------------------------------------------------------

def print_container_reference() -> None:
    """
    Print the docker run commands and key parameters for Azure AI containers.
    These are the commands you would run in a real deployment.
    """
    print("\n" + "=" * 70)
    print("AZURE AI SERVICES CONTAINER REFERENCE")
    print("=" * 70)

    print("""
All Azure AI containers require THREE mandatory parameters:
  1. Eula=accept          - Accept the license agreement
  2. Billing=<endpoint>   - Your Azure resource endpoint (for billing telemetry)
  3. ApiKey=<key>         - Your Azure resource API key

Containers process requests locally but send billing data to Azure.
No data content is sent to Azure - only billing metrics.
""")

    # Language container
    print("--- Language / Text Analytics Container ---")
    print(f"""
docker run --rm -it \\
  -p {LOCAL_PORT}:5000 \\
  --name {CONTAINER_NAME} \\
  mcr.microsoft.com/azure-cognitive-services/textanalytics/language:latest \\
  Eula=accept \\
  Billing={AZURE_ENDPOINT} \\
  ApiKey={AZURE_API_KEY[:8]}...{AZURE_API_KEY[-4:]}
""")

    # Vision Read container
    print("--- Vision Read (OCR) Container ---")
    print("""
docker run --rm -it \\
  -p 5001:5000 \\
  --name ai102-vision-read \\
  mcr.microsoft.com/azure-cognitive-services/vision/read:latest \\
  Eula=accept \\
  Billing=https://your-vision-resource.cognitiveservices.azure.com/ \\
  ApiKey=your-vision-api-key
""")

    # Speech-to-Text container
    print("--- Speech-to-Text Container ---")
    print("""
docker run --rm -it \\
  -p 5002:5000 \\
  --name ai102-speech-stt \\
  mcr.microsoft.com/azure-cognitive-services/speechservices/speech-to-text:latest \\
  Eula=accept \\
  Billing=https://your-speech-resource.cognitiveservices.azure.com/ \\
  ApiKey=your-speech-api-key \\
  Logging__Console__LogLevel__Default=Information
""")

    print("--- Key Container Facts for AI-102 Exam ---")
    print("""
  - Container endpoints have the SAME REST API as cloud endpoints
  - Use http://localhost:<port> instead of https://region.api.cognitive.microsoft.com
  - Health check: GET http://localhost:<port>/status
  - Containers require internet connectivity for billing (not for processing)
  - Containers do NOT send request/response data to Azure
  - MCR = Microsoft Container Registry (mcr.microsoft.com)
  - Some containers require specific SKU (e.g., S tier for certain features)
""")


# ---------------------------------------------------------------------------
# Section 2: Check if Docker is available
# ---------------------------------------------------------------------------

def check_docker_available() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  Docker version: {result.stdout.strip()}")
            return True
        else:
            print(f"  Docker not running: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  Docker not installed (not found in PATH)")
        return False
    except subprocess.TimeoutExpired:
        print("  Docker check timed out")
        return False


# ---------------------------------------------------------------------------
# Section 3: Pull the container image
# ---------------------------------------------------------------------------

def pull_container_image() -> bool:
    """
    Pull the Azure AI Language container image from MCR.
    This may take several minutes on first run (image is ~1-2 GB).
    """
    print(f"\n  Pulling image: {CONTAINER_IMAGE}")
    print("  This may take several minutes...")

    try:
        result = subprocess.run(
            ["docker", "pull", CONTAINER_IMAGE],
            capture_output=True, text=True, timeout=600  # 10 min timeout
        )
        if result.returncode == 0:
            print("  [OK] Image pulled successfully")
            return True
        else:
            print(f"  [ERROR] Pull failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] Pull timed out after 10 minutes")
        return False


# ---------------------------------------------------------------------------
# Section 4: Start the container
# ---------------------------------------------------------------------------

def start_container() -> bool:
    """
    Start the Azure AI Language container with required parameters.
    The three mandatory parameters: Eula, Billing, ApiKey.
    """
    print(f"\n  Starting container: {CONTAINER_NAME}")
    print(f"  Port mapping: localhost:{LOCAL_PORT} -> container:5000")

    # Stop existing container if running
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True, timeout=30)

    cmd = [
        "docker", "run",
        "--detach",                    # Run in background
        "-p", f"{LOCAL_PORT}:5000",   # Port mapping
        "--name", CONTAINER_NAME,
        "--memory", "4g",             # Recommended minimum for language container
        "--cpus", "1",                # CPU allocation
        CONTAINER_IMAGE,
        f"Eula=accept",               # Required: accept license
        f"Billing={AZURE_ENDPOINT}",   # Required: your Azure endpoint for billing
        f"ApiKey={AZURE_API_KEY}",     # Required: your API key
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            container_id = result.stdout.strip()[:12]
            print(f"  [OK] Container started: {container_id}")
            return True
        else:
            print(f"  [ERROR] Container failed to start: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] Container start timed out")
        return False


# ---------------------------------------------------------------------------
# Section 5: Wait for container health check
# ---------------------------------------------------------------------------

def wait_for_container_ready(max_wait: int = 120) -> bool:
    """
    Poll the container health endpoint until it's ready.
    Azure AI containers expose GET /status for readiness.
    """
    print(f"  Waiting for container to be ready (max {max_wait}s)...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            resp = requests.get(
                f"{LOCAL_CONTAINER_URL}/status",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                print(f"  Container status: {status}")
                if status == "ready":
                    print("  [OK] Container is ready!")
                    return True
        except requests.exceptions.ConnectionError:
            pass  # Container still starting
        except Exception as e:
            pass

        print("  Still starting...", end="\r")
        time.sleep(5)

    print("\n  [TIMEOUT] Container did not become ready in time")
    return False


# ---------------------------------------------------------------------------
# Section 6: Call the local container endpoint
# ---------------------------------------------------------------------------

def call_local_container() -> None:
    """
    Call the local container endpoint - same REST API as the cloud service.
    The only difference is the base URL (localhost vs Azure endpoint).
    """
    print("\n--- Calling Local Container Endpoint ---")
    print(f"  Base URL: {LOCAL_CONTAINER_URL}")
    print("  Note: Local container uses same REST API as cloud service")

    # Language detection - same payload as cloud API
    url = f"{LOCAL_CONTAINER_URL}/text/analytics/v3.1/languages"
    headers = {
        "Content-Type": "application/json",
        # No Ocp-Apim-Subscription-Key needed for local containers!
        # Authentication is handled at startup via ApiKey parameter
    }
    payload = {
        "documents": [
            {"id": "1", "text": "Hello, how are you?"},
            {"id": "2", "text": "Bonjour, comment allez-vous?"},
            {"id": "3", "text": "Hola, ¿cómo estás?"},
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        print("\n  Language Detection Results:")
        for doc in data["documents"]:
            lang = doc["detectedLanguage"]
            print(f"    Doc {doc['id']}: {lang['name']} ({lang['iso6391Name']}) "
                  f"confidence={lang['confidenceScore']:.2f}")

        print("\n  [OK] Local container is processing requests correctly")
        print("  Same API, same response format as Azure cloud endpoint!")

    except requests.exceptions.ConnectionError:
        print("  [INFO] Container not running - this is a simulation demo")
        print("  In real deployment, the container would respond identically to the cloud API")
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP error: {e}")


def call_local_container_with_sdk() -> None:
    """
    Call the local container using the Azure SDK.
    Point the SDK to localhost instead of the Azure endpoint.
    """
    print("\n--- Calling Local Container with Azure SDK ---")

    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    # For local containers, use a dummy key (auth happens at startup)
    LOCAL_ENDPOINT = LOCAL_CONTAINER_URL
    DUMMY_KEY = "any-value-works-for-local-container"  # Container ignores SDK key

    client = TextAnalyticsClient(
        endpoint=LOCAL_ENDPOINT,
        credential=AzureKeyCredential(DUMMY_KEY),
    )

    try:
        results = client.detect_language(
            ["Hello world!", "Bonjour le monde!", "Hola mundo!"]
        )
        print("  SDK Language Detection Results:")
        for i, result in enumerate(results):
            lang = result.primary_language
            print(f"    [{i+1}] {lang.name} ({lang.iso6391_name}) "
                  f"confidence={lang.confidence_score:.2f}")
        print("  [OK] SDK works with local container endpoint!")

    except Exception as e:
        print(f"  [INFO] Container not running: {type(e).__name__}")
        print("  When running, the SDK works identically with local and cloud endpoints")


# ---------------------------------------------------------------------------
# Section 7: Stop the container
# ---------------------------------------------------------------------------

def stop_container() -> None:
    """Stop and remove the demo container."""
    print(f"\n  Stopping container: {CONTAINER_NAME}")
    result = subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print("  [OK] Container stopped and removed")
    else:
        print(f"  Container stop: {result.stderr.strip() or 'already stopped'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Azure AI Services Container Deployment Demo")
    print("=" * 70)

    # Always show reference docs (no Docker needed)
    print_container_reference()

    # Check if Docker is available
    print("\n--- Checking Docker Installation ---")
    docker_available = check_docker_available()

    if docker_available:
        print("\n--- Running Container Demo ---")
        print("  Note: First pull may take 5-10 minutes (image ~1.5 GB)")

        pull_ok = pull_container_image()

        if pull_ok:
            start_ok = start_container()

            if start_ok:
                ready = wait_for_container_ready(max_wait=120)

                if ready:
                    call_local_container()
                    call_local_container_with_sdk()
                    stop_container()
                else:
                    print("  Container did not become ready - check logs: docker logs", CONTAINER_NAME)
            else:
                print("  Could not start container")
        else:
            print("  Could not pull image - check network and MCR access")
    else:
        print("\n--- Docker Not Available - Showing Simulation ---")
        print("  Install Docker Desktop to run the full demo")
        print("  https://www.docker.com/products/docker-desktop")
        call_local_container()   # Will show 'not running' gracefully
        call_local_container_with_sdk()  # Will show 'not running' gracefully

    print("\n" + "=" * 70)
    print("Container Deployment Demo Complete")
    print("=" * 70)
