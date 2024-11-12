import numpy as np
import pandas as pd

MISSING_VALUE = 2


def remove_missing(df, missing_value=MISSING_VALUE):
    """
    Remove missing values from eye-tracking data based on a specified missing value.
    Assumes 'avg_x', 'avg_y', and 'time' columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eye-tracking data with columns 'avg_x', 'avg_y', and 'time'.
    missing_value : scalar, optional
        The value representing missing data (default is 2).

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values removed.
    """
    # Remove rows where either avg_x or avg_y is equal to missing_value
    return df[(df["avg_x"] != missing_value) & (df["avg_y"] != missing_value)]


def detect_blinks(df, missing=0.0, minlen=10):
    """
    Detect blinks from eye-tracking data based on missing data points.
    A blink is defined as a continuous period during which both avg_x and avg_y are equal to the `missing` value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eye-tracking data with columns 'avg_x', 'avg_y', and 'time'.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    minlen : int, optional
        The minimum duration (in milliseconds) a period of missing data must last to be considered a blink.

    Returns
    -------
    list of dict
        A list of detected blinks, where each blink is represented as a dictionary
        containing 'start_time', 'end_time', and 'duration'.
    """
    # Mark missing data as blink events
    blink_mask = (df["avg_x"] == missing) & (df["avg_y"] == missing)
    df["is_blink"] = blink_mask

    blinks = []
    current_blink = None

    for i in range(len(df)):
        if df["is_blink"].iloc[i]:
            if current_blink is None:
                current_blink = {"start_time": df["time"].iloc[i]}
        elif current_blink is not None:
            current_blink["end_time"] = df["time"].iloc[i - 1]
            current_blink["duration"] = current_blink["end_time"] - current_blink["start_time"]
            if current_blink["duration"] >= minlen:
                blinks.append(current_blink)
            current_blink = None

    # Check if there's an ongoing blink at the end
    if current_blink is not None:
        current_blink["end_time"] = df["time"].iloc[-1]
        current_blink["duration"] = current_blink["end_time"] - current_blink["start_time"]
        if current_blink["duration"] >= minlen:
            blinks.append(current_blink)

    return blinks


def blink_rate(blinks, total_time):
    """
    Calculate the blink rate over a given time period.

    Parameters
    ----------
    blinks : list of dict
        A list of dictionaries containing blink event data.
    total_time : float
        Total observation time in seconds.

    Returns
    -------
    float
        The blink rate (blinks per minute).
    """
    blink_count = len(blinks)
    return (blink_count / total_time) * 60 if total_time > 0 else 0


def detect_fixations(df, missing=0.0, maxdist=25, mindur=50):
    """
    Identify eye fixations from eye-tracking data based on spatial and temporal criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eye-tracking data with columns 'avg_x', 'avg_y', and 'time'.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    maxdist : float, optional
        The maximum distance (in pixels) between consecutive points to be
        considered part of the same fixation (default is 25).
    mindur : int, optional
        The minimum duration (in milliseconds) a sequence of points must last
        to be considered a fixation (default is 50).

    Returns
    -------
    list of dict
        A list of fixations, each represented by a dictionary with
        'start_time', 'end_time', 'duration', 'x_mean', 'y_mean', and 'count'.
    """
    # Filter out missing data
    df_cleaned = df[(df["avg_x"] != missing) & (df["avg_y"] != missing)]

    # Calculate the distance between consecutive points
    distances = np.sqrt(np.diff(df_cleaned["avg_x"]) ** 2 + np.diff(df_cleaned["avg_y"]) ** 2)

    fixations = []
    current_fixation = []

    for i in range(len(df_cleaned)):
        if i == 0 or (len(current_fixation) > 0 and distances[i - 1] <= maxdist):
            current_fixation.append(df_cleaned.iloc[i])
        elif len(current_fixation) > 0:
            start_time = current_fixation[0]["time"]
            end_time = current_fixation[-1]["time"]
            duration = end_time - start_time
            if duration >= mindur:
                fixations.append(current_fixation)
            current_fixation = [df_cleaned.iloc[i]]
        else:
            current_fixation = [df_cleaned.iloc[i]]

    # Check if there's an ongoing fixation at the end
    if len(current_fixation) > 0:
        start_time = current_fixation[0]["time"]
        end_time = current_fixation[-1]["time"]
        duration = end_time - start_time
        if duration >= mindur:
            fixations.append(current_fixation)

    fixation_results = []
    for fixation in fixations:
        fixation_data = pd.DataFrame(fixation)
        fixation_results.append(
            {
                "start_time": fixation_data["time"].iloc[0],
                "end_time": fixation_data["time"].iloc[-1],
                "duration": fixation_data["time"].iloc[-1] - fixation_data["time"].iloc[0],
                "x_mean": fixation_data["avg_x"].mean(),
                "y_mean": fixation_data["avg_y"].mean(),
                "count": len(fixation_data),
            },
        )

    return fixation_results


def detect_saccades(df, missing=0.0, minlen=5, maxvel=40, maxacc=340):
    """
    Detect saccades from eye-tracking data based on velocity and acceleration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eye-tracking data with columns 'avg_x', 'avg_y', and 'time'.
    missing : float, optional
        The value to be treated as missing data (default is 0.0).
    minlen : int, optional
        The minimum duration (in milliseconds) a saccade must last to be detected (default is 5).
    maxvel : float, optional
        The maximum velocity (in pixels per millisecond) for a movement to be considered a saccade (default is 40).
    maxacc : float, optional
        The maximum acceleration (in pixels per millisecond squared) for a movement
        to be considered a saccade (default is 340).

    Returns
    -------
    list of dict
        A list of detected saccades, where each saccade is represented as a dictionary containing
        'start_time', 'end_time', 'duration', 'x_start', 'y_start', 'x_end', 'y_end'.
    """
    df_cleaned = df[(df["avg_x"] != missing) & (df["avg_y"] != missing)]

    if len(df_cleaned) < MISSING_VALUE:
        return []

    dx = np.diff(df_cleaned["avg_x"])
    dy = np.diff(df_cleaned["avg_y"])
    dt = np.diff(df_cleaned["time"])

    velocity = np.sqrt(dx**2 + dy**2) / dt
    acceleration = np.diff(velocity) / dt[1:]

    saccades = []
    current_saccade = None

    for i in range(len(velocity)):
        if velocity[i] > maxvel and (i == 0 or acceleration[i - 1] < maxacc):
            if current_saccade is None:
                current_saccade = {
                    "start_time": df_cleaned["time"].iloc[i],
                    "x_start": df_cleaned["avg_x"].iloc[i],
                    "y_start": df_cleaned["avg_y"].iloc[i],
                }
        elif current_saccade is not None:
            current_saccade["end_time"] = df_cleaned["time"].iloc[i - 1]
            current_saccade["duration"] = current_saccade["end_time"] - current_saccade["start_time"]
            current_saccade["x_end"] = df_cleaned["avg_x"].iloc[i - 1]
            current_saccade["y_end"] = df_cleaned["avg_y"].iloc[i - 1]
            if current_saccade["duration"] >= minlen:
                saccades.append(current_saccade)
            current_saccade = None

    if current_saccade is not None:
        current_saccade["end_time"] = df_cleaned["time"].iloc[-1]
        current_saccade["duration"] = current_saccade["end_time"] - current_saccade["start_time"]
        current_saccade["x_end"] = df_cleaned["avg_x"].iloc[-1]
        current_saccade["y_end"] = df_cleaned["avg_y"].iloc[-1]
        if current_saccade["duration"] >= minlen:
            saccades.append(current_saccade)

    return saccades
