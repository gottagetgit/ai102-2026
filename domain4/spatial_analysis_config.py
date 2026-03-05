"""
spatial_analysis_config.py
===========================
Demonstrates generating Spatial Analysis configuration for Azure Vision.
Spatial Analysis runs on an IoT Edge device (or in a container) and processes
live video streams to detect the presence and movement of people.

This script shows how to construct the JSON configuration for common
Spatial Analysis operations and provides utilities for validating and
visualising zone definitions.

Spatial Analysis Architecture:
    Camera / RTSP stream
        ↓
    Azure Vision Spatial Analysis container (Edge or Cloud)
        ↓ (CosmosDB / IoT Hub events)
    Your application

Key Concepts:
    Zone:       A polygon region drawn on the camera frame (normalised 0–1 coordinates)
    Line:       A virtual line that people can cross (for counting direction)
    Operation:  A named analysis task (e.g. count people in zone, detect crossing)
    Event:      A message fired when an operation condition is met (JSON payload)

Operations Demonstrated:
    cognitiveservices.vision.spatialanalysis-persondistance
        Detects if people are closer together than a threshold distance.
    cognitiveservices.vision.spatialanalysis-personcrossingline
        Fires an event when a person crosses a virtual line (tracks direction).
    cognitiveservices.vision.spatialanalysis-personcrossingpolygon
        Fires events when people enter or exit a zone polygon.
    cognitiveservices.vision.spatialanalysis-personcounting
        Counts people in a zone; fires when count changes or threshold met.
    cognitiveservices.vision.spatialanalysis-personzonecrossingevents
        Advanced zone crossing with dwell time and crowd metrics.

Exam Skill Mapping:
    - "Use Azure Vision in Foundry Tools Spatial Analysis to detect presence
       and movement of people in video"

No credentials required — this script generates and validates configuration JSON.
It does not make live API calls.

Install:
    pip install python-dotenv  (optional, only if integrating with other scripts)
"""

import json
import math
from typing import Any


# ===========================================================================
# COORDINATE SYSTEM
# Spatial Analysis coordinates are NORMALISED (0.0 – 1.0) relative to the
# frame dimensions. This makes configs resolution-independent.
# Point format: [x, y] where (0,0) = top-left, (1,1) = bottom-right
# ===========================================================================

# ---------------------------------------------------------------------------
# Zone helper utilities
# ---------------------------------------------------------------------------

def validate_polygon(points: list[list[float]], name: str = "zone") -> list[str]:
    """Validate a polygon definition for Spatial Analysis.

    Requirements:
        - At least 3 points (minimum triangle)
        - All coordinates in [0.0, 1.0]
        - Points should be listed in order (clockwise or counter-clockwise)
        - Polygon must be non-self-intersecting (convex or simple concave)

    Args:
        points: List of [x, y] coordinate pairs.
        name:   Zone name for error messages.

    Returns:
        List of warning/error strings (empty if valid).
    """
    issues = []

    if len(points) < 3:
        issues.append(f"{name}: Polygon must have at least 3 points (has {len(points)}).")

    for i, pt in enumerate(points):
        if len(pt) != 2:
            issues.append(f"{name}: Point {i} must be [x, y], got {pt}")
            continue
        x, y = pt
        if not (0.0 <= x <= 1.0):
            issues.append(f"{name}: Point {i} x={x} is outside [0, 1].")
        if not (0.0 <= y <= 1.0):
            issues.append(f"{name}: Point {i} y={y} is outside [0, 1].")

    return issues


def polygon_area(points: list[list[float]]) -> float:
    """Compute the area of a polygon using the Shoelace formula (normalised units)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def pixel_polygon(norm_points: list[list[float]], width: int, height: int) -> list[list[int]]:
    """Convert normalised polygon to pixel coordinates for a given frame size.

    Useful for debugging: draw the zone on an actual camera frame image.

    Args:
        norm_points: Normalised [x, y] pairs.
        width:       Frame width in pixels.
        height:      Frame height in pixels.

    Returns:
        List of [px, py] integer pixel coordinate pairs.
    """
    return [
        [round(x * width), round(y * height)]
        for x, y in norm_points
    ]


# ---------------------------------------------------------------------------
# Operation configuration builders
# ---------------------------------------------------------------------------

def build_person_distance_operation(
    operation_id: str,
    zone_polygon: list[list[float]],
    min_distance_threshold: float = 1.5,
    focus: str = "feet",
) -> dict:
    """Build a personDistancing operation configuration.

    Fires an event when two or more people are detected closer than
    `min_distance_threshold` metres.

    Args:
        operation_id:            Unique ID for this operation.
        zone_polygon:            List of [x,y] normalised points defining the area.
        min_distance_threshold:  Alert when people are within this distance (metres).
        focus:                   Body keypoint used for distance measurement.
                                 "feet" (default), "head", or "body"

    Returns:
        Operation config dict.
    """
    return {
        "type": "Microsoft.VideoAnalytics.SpatialAnalysis",
        "nodes": [
            {
                "name": operation_id,
                "type": "Microsoft.VideoAnalytics.SpatialAnalysis.PersonDistance",
                "configurations": {
                    "zones": [
                        {
                            "name":    f"{operation_id}_zone",
                            "polygon": zone_polygon,
                        }
                    ],
                    "outputFrequency":        "eventDriven",
                    "minimumDistanceThreshold": min_distance_threshold,
                    "maximumDistanceThreshold": 10.0,
                    "focus":                   focus,
                    "enableFaceMaskClassifier": False,
                },
            }
        ],
    }


def build_person_crossing_line_operation(
    operation_id: str,
    line_points: list[list[float]],
    sensitivity: str = "medium",
) -> dict:
    """Build a personcrossingline operation configuration.

    Fires an event when a person crosses the specified virtual line.
    Direction is determined by which side of the line the person came from.

    Args:
        operation_id:   Unique ID for this operation.
        line_points:    Two [x,y] points defining the line: [[x1,y1],[x2,y2]].
        sensitivity:    Detection sensitivity: "low", "medium", or "high".
                        Higher sensitivity = more events but more false positives.

    Returns:
        Operation config dict.
    """
    if len(line_points) != 2:
        raise ValueError("line_points must contain exactly 2 points [[x1,y1],[x2,y2]].")

    return {
        "type": "Microsoft.VideoAnalytics.SpatialAnalysis",
        "nodes": [
            {
                "name": operation_id,
                "type": "Microsoft.VideoAnalytics.SpatialAnalysis.PersonCrossingLine",
                "configurations": {
                    "lines": [
                        {
                            "name":   f"{operation_id}_line",
                            "line":   line_points,
                            "events": [
                                {
                                    "type": "linecrossing",
                                    "config": {
                                        "trigger":     "event",
                                        "focus":       "footprint",
                                        "threshold":   0.3,
                                        "outputFrequency": 0,
                                    },
                                }
                            ],
                        }
                    ],
                    "sensitivity": sensitivity,
                },
            }
        ],
    }


def build_person_counting_operation(
    operation_id: str,
    zone_polygon: list[list[float]],
    threshold_count: int = 10,
    output_frequency: str = "1.0",
) -> dict:
    """Build a personcounting operation configuration.

    Counts the number of people in a zone. Fires an event on count changes
    and when the count crosses the threshold.

    Args:
        operation_id:     Unique ID for this operation.
        zone_polygon:     Normalised polygon points defining the counting zone.
        threshold_count:  Fire a crowding alert if count exceeds this value.
        output_frequency: Reporting interval in seconds (as a string).
                          Use "0" for event-driven only.

    Returns:
        Operation config dict.
    """
    return {
        "type": "Microsoft.VideoAnalytics.SpatialAnalysis",
        "nodes": [
            {
                "name": operation_id,
                "type": "Microsoft.VideoAnalytics.SpatialAnalysis.PersonCounting",
                "configurations": {
                    "zones": [
                        {
                            "name":    f"{operation_id}_zone",
                            "polygon": zone_polygon,
                            "events": [
                                {
                                    "type": "zonecounting",
                                    "config": {
                                        "trigger":         "interval",
                                        "output_frequency": output_frequency,
                                        "threshold":        str(threshold_count),
                                    },
                                },
                                {
                                    "type": "crowdcount",
                                    "config": {
                                        "trigger":   "threshold",
                                        "threshold": str(threshold_count),
                                        "focus":     "footprint",
                                    },
                                },
                            ],
                        }
                    ],
                    "enableFaceMaskClassifier": False,
                },
            }
        ],
    }


def build_zone_crossing_operation(
    operation_id: str,
    zone_polygon: list[list[float]],
    dwell_time_threshold: float = 30.0,
) -> dict:
    """Build a personzonecrossing operation with dwell time tracking.

    This advanced operation tracks:
        - Zone entry events
        - Zone exit events
        - Dwell time (how long a person stays in the zone)

    Args:
        operation_id:          Unique ID for this operation.
        zone_polygon:          Normalised polygon points for the zone.
        dwell_time_threshold:  Alert if a person dwells longer than this (seconds).

    Returns:
        Operation config dict.
    """
    return {
        "type": "Microsoft.VideoAnalytics.SpatialAnalysis",
        "nodes": [
            {
                "name": operation_id,
                "type": "Microsoft.VideoAnalytics.SpatialAnalysis.PersonCrossingPolygon",
                "configurations": {
                    "zones": [
                        {
                            "name":    f"{operation_id}_zone",
                            "polygon": zone_polygon,
                            "events": [
                                {
                                    "type": "zonecrossing",
                                    "config": {
                                        "trigger":  "event",
                                        "focus":    "footprint",
                                    },
                                },
                                {
                                    "type": "zonedwelltime",
                                    "config": {
                                        "trigger":   "threshold",
                                        "threshold": str(dwell_time_threshold),
                                        "focus":     "footprint",
                                    },
                                },
                            ],
                        }
                    ],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Full deployment manifest builder
# ---------------------------------------------------------------------------

def build_deployment_manifest(
    camera_rtsp_url: str,
    operations: list[dict],
    iot_hub_connection_string: str = "${IOT_HUB_CONNECTION_STRING}",
    frame_rate: int = 2,
    gpu: str = "auto",
) -> dict:
    """Build a complete IoT Edge deployment manifest for Spatial Analysis.

    This is the top-level configuration deployed to an Azure IoT Edge device
    running the Spatial Analysis container module.

    Args:
        camera_rtsp_url:           RTSP stream URL (e.g. rtsp://camera-ip:554/stream1)
        operations:                List of operation config dicts.
        iot_hub_connection_string: IoT Hub device connection string (or env var reference).
        frame_rate:                Frames per second to analyse (reduce to lower cost).
        gpu:                       GPU selection: "auto", "cuda:0", or "cpu".

    Returns:
        Full deployment manifest dict suitable for iotedge deployment.
    """
    # Merge all operation nodes into a flat list
    all_nodes = []
    for op in operations:
        all_nodes.extend(op.get("nodes", []))

    manifest = {
        "version": "1.0",
        "type": "Microsoft.VideoAnalytics",
        "module": {
            "name": "spatialanalysis",
            "image": "mcr.microsoft.com/azure-cognitive-services/vision/spatial-analysis:1-amd64",
            "env": {
                "BILLING": {
                    "value": "${AZURE_AI_SERVICES_ENDPOINT}"
                },
                "APIKEY": {
                    "value": "${AZURE_AI_SERVICES_KEY}"
                },
                "EULA": {
                    "value": "accept"
                },
            },
            "settings": {
                "frameRate": frame_rate,
                "gpu":       gpu,
            },
        },
        "pipeline": {
            "nodes": [
                # RTSP source node — connects to the camera
                {
                    "name": "camera_source",
                    "type": "Microsoft.VideoAnalytics.CameraSource",
                    "configurations": {
                        "url":       camera_rtsp_url,
                        "frameRate": frame_rate,
                    },
                },
                # Operation nodes (generated above)
                *all_nodes,
                # IoT Hub sink — sends events to IoT Hub
                {
                    "name": "iothub_sink",
                    "type": "Microsoft.VideoAnalytics.IotHubMessageSink",
                    "configurations": {
                        "iotHubConnectionString": iot_hub_connection_string,
                    },
                },
            ]
        },
    }
    return manifest


def validate_deployment_manifest(manifest: dict) -> list[str]:
    """Perform basic validation checks on a deployment manifest.

    Args:
        manifest: Deployment manifest dict.

    Returns:
        List of issue strings (empty if all checks pass).
    """
    issues = []

    # Check required top-level fields
    for field in ("version", "type", "module", "pipeline"):
        if field not in manifest:
            issues.append(f"Missing required field: '{field}'")

    # Check pipeline nodes
    pipeline = manifest.get("pipeline", {})
    nodes = pipeline.get("nodes", [])

    if not any(n.get("type") == "Microsoft.VideoAnalytics.CameraSource" for n in nodes):
        issues.append("Pipeline is missing a CameraSource node.")

    if not any("Sink" in n.get("type", "") for n in nodes):
        issues.append("Pipeline is missing an output Sink node.")

    # Check each operation node has a name
    for node in nodes:
        if not node.get("name"):
            issues.append(f"A pipeline node is missing a 'name' field: {node}")

    # Validate zone polygons
    for node in nodes:
        configs = node.get("configurations", {})
        for zone in configs.get("zones", []):
            polygon = zone.get("polygon", [])
            issues.extend(validate_polygon(polygon, zone.get("name", "unnamed_zone")))
        for line_def in configs.get("lines", []):
            line_pts = line_def.get("line", [])
            if len(line_pts) != 2:
                issues.append(
                    f"Line '{line_def.get('name')}' must have exactly 2 points; "
                    f"has {len(line_pts)}."
                )

    return issues


# ---------------------------------------------------------------------------
# Example scenario builder
# ---------------------------------------------------------------------------

def create_retail_store_scenario() -> dict:
    """Create a full Spatial Analysis configuration for a retail store.

    Scenario: Monitor a store entrance + checkout area with:
        1. People counting at the entrance
        2. Line crossing counter for entry/exit
        3. Queue monitoring at checkout
        4. Social distance monitoring in the waiting area
    """
    # Entrance counting zone (top ~30% of frame, full width)
    entrance_zone = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.30], [0.0, 0.30]]

    # Virtual entry/exit line across the entrance (10% down from top)
    entrance_line = [[0.0, 0.10], [1.0, 0.10]]

    # Checkout queue zone (bottom-right quadrant)
    checkout_zone = [[0.55, 0.55], [1.0, 0.55], [1.0, 1.0], [0.55, 1.0]]

    # Waiting area (centre of frame) for social distancing
    waiting_zone = [[0.25, 0.35], [0.75, 0.35], [0.75, 0.75], [0.25, 0.75]]

    operations = [
        build_person_counting_operation(
            "entrance_count",
            entrance_zone,
            threshold_count=20,
            output_frequency="5.0",
        ),
        build_person_crossing_line_operation(
            "entry_exit_line",
            entrance_line,
            sensitivity="medium",
        ),
        build_zone_crossing_operation(
            "checkout_queue",
            checkout_zone,
            dwell_time_threshold=120.0,  # Alert if waiting > 2 minutes
        ),
        build_person_distance_operation(
            "waiting_area_distancing",
            waiting_zone,
            min_distance_threshold=1.5,
        ),
    ]

    manifest = build_deployment_manifest(
        camera_rtsp_url="rtsp://camera.store.local:554/entrance",
        operations=operations,
        frame_rate=2,
    )

    return manifest


def create_car_park_scenario() -> dict:
    """Create a Spatial Analysis config for a car park pedestrian safety zone.

    Detects pedestrians in the vehicle lane (danger zone) and
    monitors the pedestrian crossing line.
    """
    # Danger zone: vehicle lane (full-width strip in middle of frame)
    danger_zone = [[0.0, 0.40], [1.0, 0.40], [1.0, 0.60], [0.0, 0.60]]

    # Pedestrian crossing line (vertical line, right side)
    crossing_line = [[0.80, 0.0], [0.80, 1.0]]

    operations = [
        build_zone_crossing_operation(
            "vehicle_lane_intrusion",
            danger_zone,
            dwell_time_threshold=5.0,  # Alert if in danger zone > 5 seconds
        ),
        build_person_crossing_line_operation(
            "pedestrian_crossing",
            crossing_line,
            sensitivity="high",  # Prioritise safety: accept more false positives
        ),
    ]

    return build_deployment_manifest(
        camera_rtsp_url="rtsp://camera.carpark.local:554/level1",
        operations=operations,
        frame_rate=4,  # Higher rate for safety-critical scenario
    )


def print_scenario(name: str, manifest: dict) -> None:
    """Print a scenario name, its zone summary, and the full JSON."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    nodes = manifest.get("pipeline", {}).get("nodes", [])
    print(f"\nPipeline nodes: {len(nodes)}")
    for node in nodes:
        print(f"  [{node.get('type', 'unknown').split('.')[-1]}] {node.get('name', '')}")

    print("\n--- JSON Configuration ---")
    print(json.dumps(manifest, indent=2))


def run_spatial_analysis_demo():
    """Generate and validate multiple Spatial Analysis scenarios."""
    scenarios = [
        ("Retail Store", create_retail_store_scenario()),
        ("Car Park Safety", create_car_park_scenario()),
    ]

    for scenario_name, manifest in scenarios:
        print_scenario(scenario_name, manifest)

        # Validate
        issues = validate_deployment_manifest(manifest)
        if issues:
            print(f"\n⚠ Validation issues ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✓ Validation passed — configuration is well-formed.")

        # Area stats for each zone
        print("\n--- Zone Area Statistics ---")
        for node in manifest.get("pipeline", {}).get("nodes", []):
            configs = node.get("configurations", {})
            for zone in configs.get("zones", []):
                polygon = zone.get("polygon", [])
                area = polygon_area(polygon)
                pct  = area * 100
                print(
                    f"  Zone '{zone['name']}': area = {area:.4f} normalised units "
                    f"({pct:.1f}% of frame)"
                )
            for line_def in configs.get("lines", []):
                pts = line_def.get("line", [])
                if len(pts) == 2:
                    length = math.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
                    print(f"  Line '{line_def['name']}': length = {length:.4f} normalised units")

    # Save all configs
    output = {s: m for s, m in scenarios}
    out_file = "spatial_analysis_configs.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nAll configurations saved to: {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure Vision Spatial Analysis — Configuration Demo ===\n")
    print("Note: This script generates configuration JSON only.")
    print("No Azure API calls are made.\n")

    run_spatial_analysis_demo()
