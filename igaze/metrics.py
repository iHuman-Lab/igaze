from collections import defaultdict

import numpy as np

from igaze.detectors import detect_fixations, detect_saccades
from igaze.utils import is_within_aoi, parse_fixations


def scan_path(fixations):
    """Calculate the average length of the scan path based on fixations.

    Parameters
    ----------
    fixations : list
        A list of fixation events.

    Returns
    -------
    float
        The average scan path length.
    """
    # Parse fixation data
    fix = parse_fixations(fixations)

    # Create an array of x and y coordinates
    coordinates = np.column_stack((fix["x"], fix["y"]))

    # Calculate the differences between consecutive fixations
    differences = np.diff(coordinates, axis=0)

    # Calculate the Euclidean distances and return the average
    path_lengths = np.linalg.norm(differences, axis=1)
    return np.nanmean(path_lengths) if len(path_lengths) > 0 else 0.0


def fixation_metrics(df, areas):
    """
    Analyze fixations to count the number of fixations and calculate dwell time for each defined area of interest (AOI).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw eye-tracking data with columns such as 'avg_x', 'avg_y', 'time'.
    areas : dict
        A dictionary where keys are area names and values are tuples (x_min, y_min, x_max, y_max).

    Returns
    -------
    tuple
        - A dictionary with area names as keys and the count of fixations as values.
        - A dictionary with area names as keys and total duration of fixations as values.
        - A dictionary with area names as keys and percentage of time spent as values.
    """
    # Detect fixations first
    fixations = detect_fixations(df)

    fixation_count = defaultdict(int)
    fixation_duration = defaultdict(float)
    total_duration = 0.0

    for fixation in fixations:
        total_duration += fixation["duration"]  # Accumulate the total duration

        # Iterate over all AOIs and check if the current fixation falls within any AOI
        for area, bounds in areas.items():
            if is_within_aoi(fixation, bounds):  # Use is_within_aoi to check if fixation is within the AOI
                fixation_count[area] += 1  # Increment fixation count for this area
                fixation_duration[area] += fixation["duration"]  # Add duration to the total for this area

    # Calculate percentage of time spent in each AOI
    percentage_time = {
        area: (time / total_duration * 100) if total_duration else 0 for area, time in fixation_duration.items()
    }

    return fixation_count, fixation_duration, percentage_time


def analyze_saccades(df, distance_between_eyes):
    """
    Analyze saccades to calculate angular amplitudes and frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw eye-tracking data with columns such as 'avg_x', 'avg_y', 'time'.
    distance_between_eyes : float
        The distance from the eye to the fixation plane in millimeters.

    Returns
    -------
    dict
        - 'amplitudes' : List of angular amplitudes for each saccade.
        - 'frequency' : Frequency of saccades in Hz.
    """
    # Detect saccades first
    saccades = detect_saccades(df)

    amplitudes = []
    saccades_count = len(saccades)

    for saccade in saccades:
        dx = saccade["x_end"] - saccade["x_start"]
        dy = saccade["y_end"] - saccade["y_start"]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate angular amplitude (in degrees)
        amplitude = 2 * np.arctan(distance / (2 * distance_between_eyes)) * (180 / np.pi)  # Convert radians to degrees
        amplitudes.append(amplitude)

    # Calculate saccade frequency
    if saccades_count == 0:
        frequency = 0.0
    else:
        start_time = saccades[0]["start_time"]
        end_time = saccades[-1]["end_time"]
        total_time = (end_time - start_time) / 1000.0  # Convert ms to seconds
        frequency = saccades_count / total_time if total_time > 0 else 0.0

    return {"amplitudes": amplitudes, "frequency": frequency}


def aoi_entropy(aoi_fixation_counts):
    """
    Calculate the entropy based on Area of Interest (AOI) fixation distribution.

    Parameters
    ----------
    aoi_fixation_counts : dict
        A dictionary where keys are AOIs and values are the corresponding fixation counts.

    Returns
    -------
    float
        The calculated entropy value based on the fixation distribution.
    """
    fixation_hits = sum(aoi_fixation_counts.values())
    # Calculate probabilities based on fixation counts and compute entropy
    probabilities = [count / fixation_hits for count in aoi_fixation_counts.values() if fixation_hits > 0]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def dwell_metrics(df, x_min, x_max, y_min, y_max):
    """
    Calculate the total dwell time and count of gaze points within a specified Area of Interest (AOI).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw eye-tracking data with columns like 'avg_x', 'avg_y', 'time'.
    x_min, x_max, y_min, y_max : float
        The boundaries of the AOI.

    Returns
    -------
    total_dwell_time_in_aoi : float
        Total duration spent on gaze points within the AOI.
    dwell_count : int
        Number of gaze points within the AOI.
    """
    aoi_gaze_points = df[
        (df["avg_x"] >= x_min) & (df["avg_x"] <= x_max) & (df["avg_y"] >= y_min) & (df["avg_y"] <= y_max)
    ]

    total_dwell_time_in_aoi = aoi_gaze_points["time"].sum()
    dwell_count = len(aoi_gaze_points)

    return total_dwell_time_in_aoi, dwell_count


def turn_rate(gaze_data, aoi1_bounds, aoi2_bounds):
    """
    Calculate the turn rate based on transitions between two areas of interest (AOIs).

    Parameters
    ----------
    gaze_data : pd.DataFrame
        DataFrame containing gaze positions (x, y).
    aoi1_bounds, aoi2_bounds : tuple
        Boundaries of AOI1 and AOI2.

    Returns
    -------
    int
        Total number of transitions between AOI1 and AOI2.
    """
    previous_aoi = None
    transition_count = 0

    for _, gaze_point in gaze_data.iterrows():
        current_aoi = None
        if is_within_aoi(gaze_point, aoi1_bounds):
            current_aoi = "AOI1"
        elif is_within_aoi(gaze_point, aoi2_bounds):
            current_aoi = "AOI2"

        # Count transition from one AOI to the other
        if previous_aoi and previous_aoi != current_aoi and current_aoi:
            transition_count += 1

        previous_aoi = current_aoi

    return transition_count


def gaze_hit_rate_per_aoi(participant_data, aois):
    """
    Calculate the gaze hit rate for each AOI based on participants' gaze data.

    Parameters
    ----------
    participant_data : list of dict
        List of participant gaze data where each dictionary contains 'x', 'y' coordinates.
    aois : list of dict
        List of AOIs where each dictionary defines the boundary for an AOI.

    Returns
    -------
    dict
        Dictionary of hit rate for each AOI.
    """
    hit_rates = defaultdict(int)
    total_count = len(participant_data)

    for gaze_point in participant_data:
        for aoi_name, aoi_bounds in aois.items():
            if is_within_aoi(gaze_point, aoi_bounds):
                hit_rates[aoi_name] += 1

    return {aoi: count / total_count for aoi, count in hit_rates.items()}
