"""
monitor_resource.py
===================
Demonstrates querying Azure Monitor metrics for an Azure AI Services resource.
Shows request counts, latency percentiles, and error rates using the
azure-mgmt-monitor SDK.

Exam Skill: "Monitor an Azure AI resource" (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Listing available metrics for a Cognitive Services resource
  - Querying TotalCalls, TotalErrors, and SuccessfulCalls metrics
  - Querying latency metrics with percentile aggregations
  - Filtering metrics by time range and granularity
  - Interpreting metric data for operational monitoring

Metric reference for Azure AI Services:
  https://learn.microsoft.com/azure/ai-services/metrics

Required packages:
  pip install azure-mgmt-monitor azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_SUBSCRIPTION_ID   - Your Azure subscription ID
  AZURE_RESOURCE_GROUP    - Resource group containing the AI resource
  AZURE_AI_ACCOUNT_NAME   - Name of the Cognitive Services account
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import MetricAggregationType
from azure.core.exceptions import AzureError

load_dotenv()

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP  = os.environ["AZURE_RESOURCE_GROUP"]
ACCOUNT_NAME    = os.environ["AZURE_AI_ACCOUNT_NAME"]

# Construct the full resource ID for the AI Services account
RESOURCE_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.CognitiveServices/accounts/{ACCOUNT_NAME}"
)


def get_monitor_client() -> MonitorManagementClient:
    """Create an authenticated Azure Monitor management client."""
    credential = DefaultAzureCredential()
    return MonitorManagementClient(credential, SUBSCRIPTION_ID)


def list_available_metrics(client: MonitorManagementClient) -> None:
    """
    List all metrics available for this Cognitive Services resource.
    Useful when building dashboards or alerts to know what you can monitor.
    """
    print("\n[METRICS] Available metrics for this resource:")
    metric_defs = list(client.metric_definitions.list(RESOURCE_ID))

    if not metric_defs:
        print("  No metrics found (resource may not have emitted data yet).")
        return

    for m in metric_defs:
        agg_types = [a.aggregation_type for a in (m.metric_availabilities or [])]
        print(f"  {m.name.value:40s} | Unit: {m.unit:12s} | ID: {m.id}")


def query_call_metrics(
    client: MonitorManagementClient,
    hours_back: int = 24,
) -> None:
    """
    Query total calls, successful calls, and errors over a time window.

    Key metrics for AI Services health monitoring:
      - TotalCalls: All API calls received (billable + free)
      - SuccessfulCalls: Calls that returned HTTP 2xx
      - TotalErrors: Calls that returned HTTP 4xx or 5xx
      - BlockedCalls: Calls exceeding rate limit / quota
    """
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    print(f"\n[CALLS] Call metrics for past {hours_back} hours")
    print(f"  From : {start_time.isoformat()}")
    print(f"  To   : {end_time.isoformat()}")

    metrics_to_query = "TotalCalls,SuccessfulCalls,TotalErrors,BlockedCalls"

    try:
        result = client.metrics.list(
            resource_uri=RESOURCE_ID,
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            interval="PT1H",
            metricnames=metrics_to_query,
            aggregation="Total",
        )

        for metric in result.value:
            print(f"\n  Metric: {metric.name.value}")
            total = 0
            for ts in metric.timeseries:
                for dp in ts.data:
                    value = dp.total or 0
                    total += value
                    if value > 0:
                        print(f"    {dp.time_stamp.strftime('%Y-%m-%d %H:%M')} UTC : {value:.0f}")
            print(f"    TOTAL over period: {total:.0f}")

    except AzureError as e:
        print(f"  [ERROR] Could not retrieve call metrics: {e}")


def query_latency_metrics(
    client: MonitorManagementClient,
    hours_back: int = 24,
) -> None:
    """
    Query latency metrics with percentile aggregations.
    """
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    print(f"\n[LATENCY] Latency metrics for past {hours_back} hours")

    try:
        result = client.metrics.list(
            resource_uri=RESOURCE_ID,
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            interval="PT1H",
            metricnames="LatencyE2E",
            aggregation="Average,Maximum",
        )

        for metric in result.value:
            print(f"\n  Metric: {metric.name.value} (milliseconds)")
            for ts in metric.timeseries:
                for dp in ts.data:
                    avg = dp.average
                    max_ = dp.maximum
                    if avg is not None or max_ is not None:
                        print(
                            f"    {dp.time_stamp.strftime('%Y-%m-%d %H:%M')} UTC"
                            f" | Avg: {avg:.1f} ms" if avg else ""
                            f" | Max: {max_:.1f} ms" if max_ else ""
                        )

    except AzureError as e:
        print(f"  [WARN] LatencyE2E metric not available or no data: {e}")
        print("  (This metric requires the resource to have received traffic.)")


def query_token_metrics(
    client: MonitorManagementClient,
    hours_back: int = 24,
) -> None:
    """
    Query token consumption metrics for Azure OpenAI resources.
    Only applicable if the account kind is 'OpenAI'.
    """
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    print(f"\n[TOKENS] Token metrics for past {hours_back} hours (OpenAI only)")

    try:
        result = client.metrics.list(
            resource_uri=RESOURCE_ID,
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            interval="PT1H",
            metricnames="TokenTransaction",
            aggregation="Total",
        )

        for metric in result.value:
            print(f"\n  Metric: {metric.name.value}")
            grand_total = 0
            for ts in metric.timeseries:
                for dp in ts.data:
                    value = dp.total or 0
                    grand_total += value
                    if value > 0:
                        print(f"    {dp.time_stamp.strftime('%Y-%m-%d %H:%M')} UTC : {value:.0f} tokens")
            print(f"    TOTAL tokens over period: {grand_total:.0f}")

    except AzureError as e:
        print(f"  [WARN] Token metrics not available: {e}")
        print("  (TokenTransaction is only available for Azure OpenAI accounts.)")


def show_alert_setup_guidance() -> None:
    """
    Print guidance on setting up metric alerts for AI Services.
    """
    print("\n" + "=" * 60)
    print("Alert Setup Reference (azure-mgmt-monitor AlertsManagement)")
    print("=" * 60)
    print("""
  Common alerting thresholds for Azure AI Services:

  1. High Error Rate:
       Metric    : TotalErrors
       Condition : Total > 50 in 15 minutes
       Severity  : 2 (Warning)
       Action    : Email / webhook / Logic App

  2. Quota Exhaustion:
       Metric    : BlockedCalls
       Condition : Total > 0 in 5 minutes
       Severity  : 1 (Critical) - you're being throttled

  3. High Latency:
       Metric    : LatencyE2E
       Condition : Average > 5000 ms over 30 minutes
       Severity  : 3 (Informational)

  To create an alert rule via CLI:
    az monitor metrics alert create \\
      --name "HighErrorRate" \\
      --resource-group <rg> \\
      --scopes <resource-id> \\
      --condition "total TotalErrors > 50" \\
      --window-size 15m \\
      --evaluation-frequency 5m \\
      --severity 2 \\
      --description "AI Services error rate spike"

  Via Python SDK, use MonitorManagementClient.metric_alerts.create_or_update()
""")


def main():
    print("=" * 60)
    print("Azure AI Services Monitoring Demo")
    print("=" * 60)
    print(f"Resource: {RESOURCE_ID}")

    try:
        client = get_monitor_client()
        list_available_metrics(client)
        query_call_metrics(client, hours_back=24)
        query_latency_metrics(client, hours_back=24)
        query_token_metrics(client, hours_back=24)
        show_alert_setup_guidance()

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except AzureError as e:
        print(f"\n[ERROR] Azure Monitor error: {e}")
        print("Ensure your identity has 'Monitoring Reader' role on the resource.")


if __name__ == "__main__":
    main()
