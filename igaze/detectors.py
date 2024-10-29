import numpy as np
import pandas as pd

from igaze.gazeplotter import parse_fixations

MISSING_VALUE = 2


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


def blink_rate(blinks, total_time):
    """
    Calculate the blink rate over a given time period.

    Parameters
    ----------
    blinks : list of dict
        A list of dictionaries containing blink event data,
        where each dictionary has 'start_time', 'end_time', and 'duration'.

    total_time : float
        Total observation time in seconds.

    Returns
    -------
    float
        The blink rate (blinks per minute).
    """
    blink_count = len(blinks)  # Count the number of blinks

    return (blink_count / total_time) * 60 if total_time > 0 else 0


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


def saccades(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):  # noqa: PLR0913
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


def fixation_metrics(fixations, areas):
    """
    Analyze fixations to count the number of fixations and calculate dwell time
    for each defined area of interest (AOI).

    Parameters
    ----------
    fixations : list of dict
        A list of dictionaries where each fixation contains 'x_mean', 'y_mean', and 'duration'.

    areas : dict
        A dictionary where keys are area names and values are tuples
        defining the area boundaries (x_min, y_min, x_max, y_max).

    Returns
    -------
    tuple
        A tuple containing:
        - A dictionary with area names as keys and the count of fixations as values.
        - A dictionary with area names as keys and total duration of fixations as values.
        - A dictionary with area names as keys and percentage of time spent as values.
    """
    fixation_count = {area: 0 for area in areas}
    fixation_duration = {area: 0 for area in areas}
    total_duration = 0

    # Calculate counts and durations in a single pass
    for fixation in fixations:
        x, y, duration = fixation["x_mean"], fixation["y_mean"], fixation["duration"]
        total_duration += duration

        for area, (x_min, y_min, x_max, y_max) in areas.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                fixation_count[area] += 1
                fixation_duration[area] += duration

    # Calculate percentage time spent in each area
    percentage_time = {
        area: (time / total_duration * 100) if total_duration > 0 else 0 for area, time in fixation_duration.items()
    }

    return fixation_count, fixation_duration, percentage_time


def analyze_saccades(saccades, distance_between_eyes):
    """
    Analyze saccades to calculate angular amplitudes and frequency.

    The saccade amplitude is calculated as the angular displacement
    from the initial position to the destination. The frequency is
    determined by counting the number of saccades over the total time
    interval.

    Parameters
    ----------
    saccades : list of dict
        A list of dictionaries, where each dictionary contains the following keys:
        - 'start_time' : int
            The start time of the saccade in milliseconds.
        - 'end_time' : int
            The end time of the saccade in milliseconds.
        - 'duration' : int
            The duration of the saccade in milliseconds.
        - 'x_start' : float
            The x-coordinate of the initial position in pixels.
        - 'y_start' : float
            The y-coordinate of the initial position in pixels.
        - 'x_end' : float
            The x-coordinate of the final position in pixels.
        - 'y_end' : float
            The y-coordinate of the final position in pixels.

    distance_between_eyes : float
        The distance from the eye to the fixation plane in millimeters. This
        value is crucial for calculating the angular displacement.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'amplitudes' : list of float
            A list of angular amplitudes of each saccade in degrees.
        - 'frequency' : float
            The frequency of saccades in Hz (saccades per second).

    Notes
    -----
    The angular amplitude is computed using the formula:

        Amplitude (in degrees) = 2 * arctan(d / (2 * distance_between_eyes))

    where `d` is the Euclidean distance between the start and end positions
    of the saccade.

    Example
    -------
    saccades = [
        {'start_time': 0, 'end_time': 10, 'duration': 10, 'x_start': 100,
         'y_start': 200, 'x_end': 110, 'y_end': 203},
        {'start_time': 15, 'end_time': 25, 'duration': 10, 'x_start': 110,
         'y_start': 203, 'x_end': 115, 'y_end': 210},
    ]
    distance_between_eyes = 600  # Example fixation distance
    result = analyze_saccades(saccades, distance_between_eyes)
    """
    amplitudes = []
    saccades_count = len(saccades)

    for saccade in saccades:
        # Calculate the linear distance moved (Euclidean distance)
        dx = saccade["x_end"] - saccade["x_start"]
        dy = saccade["y_end"] - saccade["y_start"]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate angular amplitude (in degrees)
        amplitude = 2 * np.arctan(distance / (2 * distance_between_eyes)) * (180 / np.pi)  # Convert radians to degrees
        amplitudes.append(amplitude)

    if saccades_count == 0:
        frequency = 0.0
    else:
        # Calculate the time span of the first and last saccade
        start_time = saccades[0]["start_time"]
        end_time = saccades[-1]["end_time"]
        total_time = (end_time - start_time) / 1000.0  # Convert ms to seconds

        frequency = saccades_count / total_time if total_time > 0 else 0.0

    return {"amplitudes": amplitudes, "frequency": frequency}
