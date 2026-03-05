"""
container_deployment.py
========================
Demonstrates pulling and running Azure AI Services containers locally
and calling the containerized endpoint.

Exam Skill: "Plan and implement a container deployment"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Understanding the Azure AI Services container model
  - Docker run commands for common AI Services containers
  - How to pass API billing configuration to containers
  - Calling the local container endpoint (same SDK, different URL)
  - Health check and readiness verification
  - Container configuration options

Azure AI Services Container Architecture:
  - Containers run the AI model locally but still require billing connectivity
  - The container sends billing telemetry to your Azure AI resource
  - API requests are processed LOCALLY (useful for data residency, offline, low-latency)
  - You still need an Azure AI Services resource for the billing endpoint and key

Container categories available (as of 2025):
  Language: Language Detection, Key Phrase Extraction, Sentiment Analysis,
            Named Entity Recognition, Text Translation
  Vision:   Read OCR, Spatial Analysis
  Speech:   Speech-to-Text, Text-to-Speech, Custom Speech
  Decision: Anomaly Detector

Required packages:
  pip install azure-ai-textanalytics requests python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT  - Billing endpoint (your cloud resource)
  AZURE_AI_SERVICES_KEY       - Billing key (your cloud resource API key)
"""

import os
import time
import subprocess
import json
import requests
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError

load_dotenv()

BILLING_ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
BILLING_KEY      = os.environ["AZURE_AI_SERVICES_KEY"]

# Local container settings
LOCAL_HOST      = "http://localhost"
LANGUAGE_PORT   = 5000   # Default port for Language service container
SPEECH_PORT     = 5001   # Default port for Speech service container
LOCAL_ENDPOINT  = f"{LOCAL_HOST}:{LANGUAGE_PORT}"

# Container images (MCR = Microsoft Container Registry)
CONTAINER_IMAGES = {
    "language_detection": "mcr.microsoft.com/azure-cognitive-services/textanalytics/language:latest",
    "sentiment":          "mcr.microsoft.com/azure-cognitive-services/textanalytics/sentiment:latest",
    "key_phrase":         "mcr.microsoft.com/azure-cognitive-services/textanalytics/keyphrase:latest",
    "ner":                "mcr.microsoft.com/azure-cognitive-services/textanalytics/ner:latest",
    "read_ocr":           "mcr.microsoft.com/azure-cognitive-services/form-recognizer/read-3.1:latest",
    "speech_to_text":     "mcr.microsoft.com/azure-cognitive-services/speechservices/speech-to-text:latest",
    "text_to_speech":     "mcr.microsoft.com/azure-cognitive-services/speechservices/text-to-speech:latest",
}

CONTAINER_NAME = "ai102-language-demo"


def check_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    print("\n[DOCKER CHECK] Verifying Docker is available...")
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  Docker version: {version}")
            print("  [OK] Docker is available.")
            return True
        else:
            print(f"  [WARN] Docker returned error: {result.stderr.strip()}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  [WARN] Docker not found or daemon not running.")
        print("  Install Docker Desktop from https://www.docker.com/products/docker-desktop")
        return False


def pull_container_image(image: str) -> bool:
    """
    Pull a container image from Microsoft Container Registry.
    Images can be large (1-4 GB) - first pull may take several minutes.
    """
    print(f"\n[PULL] Pulling image: {image}")
    print("  This may take several minutes on first run...")
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large images
        )
        if result.returncode == 0:
            print("  [OK] Image pulled successfully.")
            return True
        else:
            print(f"  [ERROR] Pull failed: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] Pull timed out after 10 minutes.")
        return False


def build_docker_run_command(
    image: str,
    port: int,
    billing_endpoint: str,
    billing_key: str,
    container_name: str,
    extra_env: dict = None,
) -> list:
    """
    Build the docker run command for an Azure AI Services container.

    Required environment variables for ALL Azure AI Services containers:
      - Eula=accept           (must accept the license agreement)
      - Billing=<endpoint>    (your Azure resource endpoint for billing)
      - ApiKey=<key>          (your Azure resource key for billing auth)

    Optional:
      - Logging:Console:LogLevel:Default=Information
      - ApplicationInsights:InstrumentationKey=<key>  (for telemetry)
    """
    cmd = [
        "docker", "run",
        "--rm",                             # Remove container when stopped
        "--name", container_name,
        "-p", f"{port}:5000",               # Map host port to container port 5000
        "-e", "Eula=accept",                # REQUIRED: accept license
        "-e", f"Billing={billing_endpoint}", # REQUIRED: billing endpoint
        "-e", f"ApiKey={billing_key}",       # REQUIRED: API key for billing
        "-e", "Logging:Console:LogLevel:Default=Information",
    ]

    # Add any extra environment variables
    if extra_env:
        for key, value in extra_env.items():
            cmd.extend(["-e", f"{key}={value}"])

    # Resource limits (recommended for production)
    cmd.extend(["--memory", "4g"])          # Language containers need ~4GB RAM
    cmd.extend(["--cpus", "2"])             # Limit CPU usage

    cmd.append(image)

    return cmd


def start_language_container(
    image: str = None,
    port: int = LANGUAGE_PORT,
) -> subprocess.Popen | None:
    """
    Start a Language Detection container in the background.
    Returns the subprocess.Popen object if started, None on failure.

    The container exposes:
      - POST /text/analytics/v3.1/languages  (Language Detection)
      - GET  /status                          (Health check)
      - GET  /swagger                         (API docs)
    """
    image = image or CONTAINER_IMAGES["language_detection"]
    cmd = build_docker_run_command(
        image=image,
        port=port,
        billing_endpoint=BILLING_ENDPOINT,
        billing_key=BILLING_KEY,
        container_name=CONTAINER_NAME,
    )

    print(f"\n[START CONTAINER]")
    print(f"  Image: {image}")
    print(f"  Port:  {port}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"  Container started with PID {process.pid}")
        return process
    except FileNotFoundError:
        print("  [ERROR] Docker not found. Cannot start container.")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to start container: {e}")
        return None


def wait_for_container_ready(
    endpoint: str = LOCAL_ENDPOINT,
    max_wait_seconds: int = 60,
) -> bool:
    """
    Poll the container's /status endpoint until it reports 'ready'.
    Language containers typically take 10-30 seconds to initialize.
    """
    print(f"\n[WAIT] Waiting for container at {endpoint} to be ready...")
    status_url = f"{endpoint}/status"
    deadline = time.time() + max_wait_seconds

    while time.time() < deadline:
        try:
            resp = requests.get(status_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                state = data.get("modelLoadingStatus", data.get("status", "unknown"))
                print(f"  Status: {state}")
                if state in ("ready", "Succeeded"):
                    print("  [OK] Container is ready to serve requests!")
                    return True
        except requests.exceptions.ConnectionError:
            pass  # Container not yet listening
        except Exception as e:
            print(f"  [WARN] Status check error: {e}")

        print(f"  Not ready yet, retrying in 3s (timeout in {int(deadline - time.time())}s)...")
        time.sleep(3)

    print(f"  [TIMEOUT] Container did not become ready within {max_wait_seconds} seconds.")
    return False


def call_local_container_endpoint(endpoint: str = LOCAL_ENDPOINT) -> None:
    """
    Call the containerized language detection service.

    The LOCAL container exposes the same REST API as the cloud service.
    You can use either:
      a) The Azure SDK pointing to the local endpoint (shown below)
      b) Direct HTTP requests to the REST API

    The SDK approach is preferred as it handles auth headers automatically.
    When using a container, the ApiKey header must still be present but
    any non-empty value is accepted (billing is handled separately).
    """
    print(f"\n[CALL CONTAINER] Calling local endpoint: {endpoint}")

    # Method A: Using the Azure SDK with local endpoint
    # The SDK sends the API key in the header but billing is handled by
    # the container's connection to your Azure resource
    client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(BILLING_KEY),
    )

    test_texts = [
        "Hello, how are you doing today?",
        "Bonjour, je voudrais un café s'il vous plaît.",
        "Hola, ¿cómo estás?",
        "こんにちは、元気ですか？",
    ]

    print(f"\n  Detecting language for {len(test_texts)} texts:")
    try:
        results = client.detect_language(test_texts)
        for text, result in zip(test_texts, results):
            if not result.is_error:
                lang = result.primary_language
                print(f"    '{text[:40]:<40}' -> {lang.name} ({lang.iso6391_name}): {lang.confidence_score:.2f}")
            else:
                print(f"    Error: {result.error.message}")
        print("  [OK] Container endpoint call successful!")
    except ServiceRequestError as e:
        print(f"  [ERROR] Connection failed - is the container running? {e}")
    except HttpResponseError as e:
        print(f"  [ERROR] API error: {e.message}")


def call_container_swagger(endpoint: str = LOCAL_ENDPOINT) -> None:
    """
    Fetch the Swagger/OpenAPI spec from the container.
    All Azure AI containers expose Swagger UI at /swagger.
    """
    print(f"\n[SWAGGER] Fetching API spec from {endpoint}/swagger")
    try:
        resp = requests.get(f"{endpoint}/swagger", timeout=10)
        if resp.status_code == 200:
            print("  Swagger UI available. Navigate to it in your browser for API documentation.")
            print(f"  URL: {endpoint}/swagger")
        else:
            print(f"  Swagger returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("  Container not running - cannot fetch Swagger.")


def stop_container(name: str = CONTAINER_NAME) -> None:
    """Stop and remove a running container by name."""
    print(f"\n[STOP] Stopping container '{name}'...")
    try:
        result = subprocess.run(
            ["docker", "stop", name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("  [OK] Container stopped.")
        else:
            print(f"  [WARN] {result.stderr.strip()}")
    except FileNotFoundError:
        print("  Docker not available.")


def show_all_container_commands() -> None:
    """
    Print ready-to-use docker run commands for popular AI Services containers.
    Useful reference for the AI-102 exam.
    """
    print("\n" + "=" * 60)
    print("Docker Run Commands Reference")
    print("=" * 60)

    containers = [
        {
            "name": "Language Detection",
            "image": CONTAINER_IMAGES["language_detection"],
            "port": 5000,
            "notes": "Detects language of input text",
        },
        {
            "name": "Sentiment Analysis v3",
            "image": CONTAINER_IMAGES["sentiment"],
            "port": 5001,
            "notes": "Positive/Negative/Neutral/Mixed sentiment",
        },
        {
            "name": "Key Phrase Extraction",
            "image": CONTAINER_IMAGES["key_phrase"],
            "port": 5002,
            "notes": "Extracts key phrases from text",
        },
        {
            "name": "Speech to Text",
            "image": CONTAINER_IMAGES["speech_to_text"],
            "port": 5003,
            "notes": "Requires Speech resource (not multi-service)",
        },
    ]

    for c in containers:
        cmd = build_docker_run_command(
            image=c["image"],
            port=c["port"],
            billing_endpoint=BILLING_ENDPOINT,
            billing_key="<YOUR_KEY>",  # Masked for display
            container_name=f"ai-{c['name'].lower().replace(' ', '-')}",
        )
        print(f"\n  {c['name']}")
        print(f"  Notes: {c['notes']}")
        print(f"  Command:")
        # Pretty print the command
        display_cmd = " \\\n    ".join(cmd)
        print(f"    {display_cmd}")


def main():
    print("=" * 60)
    print("Azure AI Services Container Deployment Demo")
    print("=" * 60)

    # 1. Check Docker availability
    docker_available = check_docker_available()

    # 2. Show all container commands (reference)
    show_all_container_commands()

    if not docker_available:
        print("\n[NOTE] Docker not available in this environment.")
        print("       The commands above show how to run containers when Docker is available.")
        print("       The demo will attempt to call a container if one is already running.")

        # Still try to call the endpoint in case a container is running
        print("\n[ATTEMPT] Checking if a container is already running on port 5000...")
        try:
            resp = requests.get(f"{LOCAL_ENDPOINT}/status", timeout=3)
            if resp.status_code == 200:
                print("  Container is already running! Calling it...")
                call_local_container_endpoint()
            else:
                print("  No container running on port 5000.")
        except requests.exceptions.ConnectionError:
            print("  No container running on port 5000.")
        return

    # 3. Pull the language detection image
    image = CONTAINER_IMAGES["language_detection"]
    pulled = pull_container_image(image)

    if not pulled:
        print("\n[SKIP] Could not pull image. Skipping container start.")
        return

    # 4. Start container
    process = start_language_container(image=image)

    if process is None:
        print("\n[SKIP] Container did not start.")
        return

    try:
        # 5. Wait for readiness
        ready = wait_for_container_ready(LOCAL_ENDPOINT, max_wait_seconds=90)

        if ready:
            # 6. Call the container
            call_local_container_endpoint()
            call_container_swagger()

    finally:
        # 7. Always clean up
        stop_container(CONTAINER_NAME)
        if process and process.poll() is None:
            process.terminate()
            print("  [CLEANUP] Container process terminated.")


if __name__ == "__main__":
    main()
