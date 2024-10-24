# -*- coding: utf-8 -*-
"""PyGazeAnalyser is a Python module for easily analysing eye-tracking data.

Copyright (C) 2014  Edwin S. Dalmaijer.
"""

import numpy as np
import pandas as pd

from igaze.gazeplotter import parse_fixations

MISSING_VALUE = 2


def detect_blinks(x, y, time, missing=0.0, minlen=10):
    """
    Detect blinks from eye-tracking data based on missing data points.

    A blink is defined as a continuous period during which both x and y coordinates
    are equal to the `missing` value for at least `minlen` milliseconds.

    Parameters
    ----------
    x : list or np.ndarray
        The x-coordinates of eye-tracking data points.
    y : list or np.ndarray
        The y-coordinates of eye-tracking data points.
    time : list or np.ndarray
        The timestamps corresponding to each (x, y) point.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    minlen : int, optional
        The minimum duration (in milliseconds) a period of missing data must last
        to be considered a blink (default is 10).

    Returns
    -------
    list of dict
        A list of detected blinks, where each blink is represented as a dictionary
        containing:
        - 'start_time' : float
            The starting timestamp of the blink.
        - 'end_time' : float
            The ending timestamp of the blink.
        - 'duration' : float
            The duration of the blink in milliseconds.

    Examples
    --------
    >>> x = [100, 100, 0, 0, 100, 100]
    >>> y = [200, 200, 0, 0, 200, 200]
    >>> time = [0, 5, 10, 15, 20, 25]
    >>> blinks = detect_blinks(x, y, time)
    >>> print(blinks)
    [{'start_time': 10, 'end_time': 15, 'duration': 5}]
    """
    # Create a DataFrame from the input data
    data = pd.DataFrame({"x": x, "y": y, "time": time})

    # Identify missing data points
    missing_mask = (data["x"] == missing) & (data["y"] == missing)
    data["is_blink"] = missing_mask

    # Initialize variables to store blinks
    blinks = []
    current_blink = None

    for i in range(len(data)):
        if data["is_blink"].iloc[i]:
            if current_blink is None:
                current_blink = {"start_time": data["time"].iloc[i]}
        elif current_blink is not None:
            current_blink["end_time"] = data["time"].iloc[i - 1]
            current_blink["duration"] = current_blink["end_time"] - current_blink["start_time"]

            if current_blink["duration"] >= minlen:
                blinks.append(current_blink)
            current_blink = None

    # Check if there's an ongoing blink at the end
    if current_blink is not None:
        current_blink["end_time"] = data["time"].iloc[-1]
        current_blink["duration"] = current_blink["end_time"] - current_blink["start_time"]

        if current_blink["duration"] >= minlen:
            blinks.append(current_blink)

    return blinks


def remove_missing(x, y, time, missing):
    """
    Remove missing values from x, y, and time arrays based on a specified missing value.

    Parameters
    ----------
    x : np.ndarray
        An array of x coordinates.

    y : np.ndarray
        An array of y coordinates.

    time : np.ndarray
        An array of time values.

    missing : scalar
        The value representing missing data.

    Returns
    -------
    tuple
        A tuple containing:
        - x_cleaned : np.ndarray
            The x coordinates with missing values removed.
        - y_cleaned : np.ndarray
            The y coordinates with missing values removed.
        - time_cleaned : np.ndarray
            The time values with missing entries removed.
    """
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    x = x[(mx + my) != MISSING_VALUE]
    y = y[(mx + my) != MISSING_VALUE]
    time = time[(mx + my) != MISSING_VALUE]
    return x, y, time


def find_fixations(x, y, time, missing=0.0, maxdist=25, mindur=50):  # noqa: PLR0913
    """
    Identify eye fixations from eye-tracking data based on spatial and temporal criteria.

    A fixation is defined as a sequence of eye-tracking points that are close together
    (within `maxdist`) and have a duration longer than `mindur`.

    Parameters
    ----------
    x : list or np.ndarray
        The x-coordinates of eye-tracking data points.
    y : list or np.ndarray
        The y-coordinates of eye-tracking data points.
    time : list or np.ndarray
        The timestamps corresponding to each (x, y) point.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    maxdist : float, optional
        The maximum distance (in pixels) between consecutive points to be considered part
        of the same fixation (default is 25).
    mindur : int, optional
        The minimum duration (in milliseconds) a sequence of points must last to be considered
        a fixation (default is 50).

    Returns
    -------
    list of dict
        A list of fixations, where each fixation is represented as a dictionary containing:
        - 'start_time' : float
            The starting timestamp of the fixation.
        - 'end_time' : float
            The ending timestamp of the fixation.
        - 'duration' : float
            The duration of the fixation in milliseconds.
        - 'x_mean' : float
            The mean x-coordinate of the fixation.
        - 'y_mean' : float
            The mean y-coordinate of the fixation.
        - 'count' : int
            The number of points in the fixation.

    Examples
    --------
    >>> x = [100, 102, 105, 300, 310, 100]
    >>> y = [200, 202, 203, 205, 210, 200]
    >>> time = [0, 10, 20, 50, 60, 100]
    >>> fixations = find_fixations(x, y, time)
    >>> print(fixations)
    [{'start_time': 0, 'end_time': 20, 'duration': 20, 'x_mean': 102.33, 'y_mean': 201.67, 'count': 3},
     {'start_time': 50, 'end_time': 100, 'duration': 50, 'x_mean': 305.0, 'y_mean': 207.5, 'count': 2}]
    """
    # Create a DataFrame from the input data
    data = pd.DataFrame({"x": x, "y": y, "time": time})

    # Filter out missing data
    data = data[(data["x"] != missing) & (data["y"] != missing)]

    # Calculate the distance between consecutive points
    distances = np.sqrt(np.diff(data["x"]) ** 2 + np.diff(data["y"]) ** 2)

    # Initialize lists to store fixations
    fixations = []
    current_fixation = []

    for i in range(len(data)):
        if i == 0 or (len(current_fixation) > 0 and distances[i - 1] <= maxdist):
            current_fixation.append(data.iloc[i])
        elif len(current_fixation) > 0:
            # Check if the current fixation has sufficient duration
            start_time = current_fixation[0]["time"]
            end_time = current_fixation[-1]["time"]
            duration = end_time - start_time

            if duration >= mindur:
                fixations.append(current_fixation)

            current_fixation = [data.iloc[i]]
        else:
            current_fixation = [data.iloc[i]]

    # Check if there's an ongoing fixation at the end
    if len(current_fixation) > 0:
        start_time = current_fixation[0]["time"]
        end_time = current_fixation[-1]["time"]
        duration = end_time - start_time

        if duration >= mindur:
            fixations.append(current_fixation)

    # Convert fixations to a more readable format
    fixation_results = []
    for fixation in fixations:
        fixation_data = pd.DataFrame(fixation)
        fixation_results.append(
            {
                "start_time": fixation_data["time"].iloc[0],
                "end_time": fixation_data["time"].iloc[-1],
                "duration": fixation_data["time"].iloc[-1] - fixation_data["time"].iloc[0],
                "x_mean": fixation_data["x"].mean(),
                "y_mean": fixation_data["y"].mean(),
                "count": len(fixation_data),
            },
        )

    return fixation_results


def calculate_saccades(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):  # noqa: PLR0913
    """
    Detect saccades from eye-tracking data based on velocity and acceleration.

    A saccade is defined as a rapid eye movement characterized by high velocity
    and acceleration, with duration and movement parameters defined by the user.

    Parameters
    ----------
    x : list or np.ndarray
        The x-coordinates of eye-tracking data points.
    y : list or np.ndarray
        The y-coordinates of eye-tracking data points.
    time : list or np.ndarray
        The timestamps corresponding to each (x, y) point.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    minlen : int, optional
        The minimum duration (in milliseconds) a saccade must last to be detected
        (default is 5).
    maxvel : float, optional
        The maximum velocity (in pixels per millisecond) for a movement to be
        considered a saccade (default is 40).
    maxacc : float, optional
        The maximum acceleration (in pixels per millisecond squared) for a movement
        to be considered a saccade (default is 340).

    Returns
    -------
    list of dict
        A list of detected saccades, where each saccade is represented as a dictionary
        containing:
        - 'start_time' : float
            The starting timestamp of the saccade.
        - 'end_time' : float
            The ending timestamp of the saccade.
        - 'duration' : float
            The duration of the saccade in milliseconds.
        - 'x_start' : float
            The starting x-coordinate of the saccade.
        - 'y_start' : float
            The starting y-coordinate of the saccade.
        - 'x_end' : float
            The ending x-coordinate of the saccade.
        - 'y_end' : float
            The ending y-coordinate of the saccade.

    Examples
    --------
    >>> x = [100, 105, 110, 300, 305, 100]
    >>> y = [200, 202, 203, 210, 215, 200]
    >>> time = [0, 5, 10, 50, 55, 100]
    >>> saccades = calculate_saccades(x, y, time)
    >>> print(saccades)
    [{'start_time': 0, 'end_time': 10, 'duration': 10, 'x_start': 100, 'y_start': 200, 'x_end': 110, 'y_end': 203}]
    """
    # Create a DataFrame from the input data
    data = pd.DataFrame({"x": x, "y": y, "time": time})

    # Filter out missing data
    data = data[(data["x"] != missing) & (data["y"] != missing)]

    if len(data) < 2:  # noqa: PLR2004
        return []  # Not enough data to calculate saccades

    # Calculate displacements and time differences
    dx = np.diff(data["x"])
    dy = np.diff(data["y"])
    dt = np.diff(data["time"])

    # Calculate velocity and acceleration
    velocity = np.sqrt(dx**2 + dy**2) / dt  # pixels per millisecond
    acceleration = np.diff(velocity) / dt[1:]  # pixels per millisecond squared

    # Initialize lists to store saccades
    saccades = []
    current_saccade = None

    for i in range(len(velocity)):
        if velocity[i] > maxvel and (i == 0 or acceleration[i - 1] < maxacc):
            if current_saccade is None:
                current_saccade = {
                    "start_time": data["time"].iloc[i],
                    "x_start": data["x"].iloc[i],
                    "y_start": data["y"].iloc[i],
                }
        elif current_saccade is not None:
            current_saccade["end_time"] = data["time"].iloc[i - 1]
            current_saccade["duration"] = current_saccade["end_time"] - current_saccade["start_time"]
            current_saccade["x_end"] = data["x"].iloc[i - 1]
            current_saccade["y_end"] = data["y"].iloc[i - 1]

            if current_saccade["duration"] >= minlen:
                saccades.append(current_saccade)
            current_saccade = None

    # Check for an ongoing saccade at the end of the data
    if current_saccade is not None:
        current_saccade["end_time"] = data["time"].iloc[-1]
        current_saccade["duration"] = current_saccade["end_time"] - current_saccade["start_time"]
        current_saccade["x_end"] = data["x"].iloc[-1]
        current_saccade["y_end"] = data["y"].iloc[-1]

        if current_saccade["duration"] >= minlen:
            saccades.append(current_saccade)

    return saccades


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


def count_fixations_per_area(fixations, areas):
    """
    Count the number of fixations for each defined area.

    Parameters
    ----------
    fixations : list
        A list of fixations, where each fixation is represented as
        a dictionary with 'coordinates' (x, y).

    areas : dict
        A dictionary where keys are area names and values are tuples
        defining the area boundaries (x_min, y_min, x_max, y_max).

    Returns
    -------
    dict
        A dictionary with area names as keys and the count of
        fixations as values.
    """
    fixation_count = {area: 0 for area in areas}

    for fixation in fixations:
        x, y = fixation["x_mean"], fixation["y_mean"]
        for area, (x_min, y_min, x_max, y_max) in areas.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                fixation_count[area] += 1

    return fixation_count
