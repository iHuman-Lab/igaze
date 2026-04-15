import numpy as np


def is_within_aoi(gaze_point, aoi):
    """
    Check if a gaze point is within the specified AOI.

    Parameters:
    gaze_point (tuple): (x, y) coordinates of the gaze point.
    aoi (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max' defining the AOI boundaries.

    Returns:
    bool: True if gaze point is within the AOI, False otherwise.
    """
    x, y = gaze_point
    return aoi["x_min"] <= x <= aoi["x_max"] and aoi["y_min"] <= y <= aoi["y_max"]


def parse_fixations(fixations):
    """
    Extract fixation data into a structured format.

    Parameters
    ----------
    fixations : list of dict
        A list of dictionaries containing fixation data.

    Returns
    -------
    dict
        A dictionary with arrays of x_mean, y_mean, and duration.
    """
    return {
        "x_mean": np.array([f["x_mean"] for f in fixations]),
        "y_mean": np.array([f["y_mean"] for f in fixations]),
        "duration": np.array([f["duration"] for f in fixations]),
    }
