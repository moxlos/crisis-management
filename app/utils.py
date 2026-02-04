#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for the Crisis Management System.

This module contains common functions used across multiple modules,
including geographical calculations.
"""

from math import sin, cos, sqrt, atan2, radians
from typing import Tuple

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6373.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.

    The Haversine formula determines the shortest distance over the earth's surface,
    giving an 'as-the-crow-flies' distance between the points.

    Formula:
        a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        c = 2 × atan2(√a, √(1−a))
        d = R × c

    Where:
        - lat1, lon1: Coordinates of point 1 (in degrees)
        - lat2, lon2: Coordinates of point 2 (in degrees)
        - R: Earth's radius (6373.0 km)
        - d: Distance between the two points (in km)

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance between the two points in kilometers

    Example:
        >>> haversine_distance(39.5534, 21.7598, 39.6234, 21.8598)
        10.23  # approximately 10.23 km
    """
    # Convert degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance
    distance = EARTH_RADIUS_KM * c
    return distance


# Alias for backward compatibility
calculate_distance = haversine_distance


# Severity level constants
SEVERITY_LEVELS = {
    1: 'Critical',
    2: 'Urgent',
    3: 'Normal'
}


def get_severity_label(severity: int) -> str:
    """Get human-readable label for severity level.

    Args:
        severity: Severity level (1, 2, or 3)

    Returns:
        Human-readable severity label
    """
    return SEVERITY_LEVELS.get(severity, 'Unknown')
