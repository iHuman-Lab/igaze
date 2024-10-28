from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# COLOURS (from Tango color palette)
COLS = {
    "butter": ["#fce94f", "#edd400", "#c4a000"],
    "orange": ["#fcaf3e", "#f57900", "#ce5c00"],
    "chocolate": ["#e9b96e", "#c17d11", "#8f5902"],
    "chameleon": ["#8ae234", "#73d216", "#4e9a06"],
    "skyblue": ["#729fcf", "#3465a4", "#204a87"],
    "plum": ["#ad7fa8", "#75507b", "#5c3566"],
    "scarletred": ["#ef2929", "#cc0000", "#a40000"],
    "aluminium": ["#eeeeec", "#d3d7cf", "#babdb6", "#888a85", "#555753", "#2e3436"],
}


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


def draw_display(size, imagefile=None):
    """
    Create a figure with a black background, optionally overlaying an image.

    Parameters
    ----------
    size : tuple of int
        Width and height of the display in pixels (width, height).
    imagefile : str, optional
        Path to the image file to overlay.

    Returns
    -------
    tuple
        A tuple containing the figure and axes objects.
    """
    screen = np.zeros((size[1], size[0], 3), dtype="uint8")

    if imagefile and Path(imagefile).is_file():
        img = Image.open(imagefile).resize(size)
        img = np.array(img)[:, :, :3][::-1]  # Keep RGB channels and flip vertically
        h, w = img.shape[:2]
        screen[(size[1] - h) // 2 : (size[1] + h) // 2, (size[0] - w) // 2 : (size[0] + w) // 2] = img
    elif imagefile:
        raise FileNotFoundError(f"Image file not found at '{imagefile}'")

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.imshow(screen, origin="upper")
    ax.axis("off")
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)  # Invert y-axis
    return fig, ax


def draw_circles(ax, x, y, sizes, colors, alpha):  # noqa: PLR0913
    """
    Draw circles on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x : array-like
        x-coordinates of the circle centers.
    y : array-like
        y-coordinates of the circle centers.
    sizes : array-like
        Sizes of the circles.
    colors : array-like
        Colors of the circles.
    alpha : float
        Transparency level of the circles.
    """
    ax.scatter(x, y, s=sizes, c=colors, marker="o", alpha=alpha, edgecolors="none")


def calculate_circle_params(fixations, size_by_duration, color_by_duration):
    """
    Calculate sizes and colors for circles based on fixation data.

    Parameters
    ----------
    fixations : dict
        A dictionary with fixation data.
    size_by_duration : bool
        Whether to size circles by duration.
    color_by_duration : bool
        Whether to color circles by duration.

    Returns
    -------
    tuple
        Sizes and colors for the circles.
    """
    sizes = fixations["duration"] / 30.0 if size_by_duration else np.median(fixations["duration"]) / 30.0
    colors = fixations["duration"] if color_by_duration else COLS["chameleon"][2]  # Default color
    return sizes, colors


def draw_fixations(  # noqa: PLR0913
    fixations,
    size,
    imagefile=None,
    alpha=0.5,
    savefile=None,
    size_by_duration=None,
    color_by_duration=None,
):
    """
    Draw circles on fixation locations, optionally overlaying an image.

    Parameters
    ----------
    fixations : list of dict
        A list of dictionaries containing fixation data.
    size : tuple of int
        Width and height of the display in pixels (width, height).
    imagefile : str, optional
        Path to the image file to overlay.
    size_by_duration : bool, optional
        Whether to size circles by duration.
    color_by_duration : bool, optional
        Whether to color circles by duration.
    alpha : float, optional
        Transparency level of the circles.
    savefile : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fix = parse_fixations(fixations)
    fig, ax = draw_display(size, imagefile=imagefile)

    sizes, colors = calculate_circle_params(fix, size_by_duration, color_by_duration)
    draw_circles(ax, fix["x_mean"], fix["y_mean"], sizes, colors, alpha)

    if savefile:
        fig.savefig(savefile)

    return fig


def create_heatmap(fixations, size, gwh=200):
    """
    Create a heatmap based on fixation data.

    Parameters
    ----------
    fixations : dict
        A dictionary with fixation data.
    size : tuple of int
        Width and height of the display in pixels (width, height).
    gwh : int, optional
        Size of the Gaussian window (default is 200).

    Returns
    -------
    np.ndarray
        The generated heatmap.
    """
    gaus = gaussian(gwh, gwh / 6)
    heatmap = np.zeros((size[1] + gwh, size[0] + gwh), dtype=float)

    for x, y, duration in zip(fixations["x"], fixations["y"], fixations["dur"]):
        x_center, y_center = int(x) + gwh // 2, int(y) + gwh // 2
        x_start, x_end = max(0, x_center - gwh // 2), min(size[0] + gwh, x_center + gwh // 2)
        y_start, y_end = max(0, y_center - gwh // 2), min(size[1] + gwh, y_center + gwh // 2)

        heatmap[y_start:y_end, x_start:x_end] += (
            gaus[
                (y_start - (y_center - gwh // 2)) : (y_end - (y_center - gwh // 2)),
                (x_start - (x_center - gwh // 2)) : (x_end - (x_center - gwh // 2)),
            ]
            * duration
        )

    return heatmap[gwh // 2 : size[1] + gwh // 2, gwh // 2 : size[0] + gwh // 2]


def draw_heatmap(fixations, size, ax, imagefile=None, alpha=0.5, savefile=None):  # noqa: PLR0913
    """
    Draw a heatmap of fixations, optionally overlaying an image.

    Parameters
    ----------
    fixations : dict
        A dictionary with fixation data.
    size : tuple of int
        Width and height of the display in pixels (width, height).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    imagefile : str, optional
        Path to the image file to overlay.
    alpha : float, optional
        Transparency level of the heatmap.
    savefile : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fix = parse_fixations(fixations)
    fig, ax = draw_display(size, imagefile=imagefile)

    heatmap = create_heatmap(fix, size)
    heatmap[heatmap < np.mean(heatmap[heatmap > 0])] = np.NaN

    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.invert_yaxis()

    if savefile:
        fig.savefig(savefile)

    return fig


def draw_eye_heatmap(fixations, size, ax, alpha=0.5):
    """
    Draw a heatmap of provided fixations.

    Parameters
    ----------
    fixations : dict
        A dictionary with fixation data.
    size : tuple of int
        Width and height of the display in pixels (width, height).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    alpha : float, optional
        Transparency level of the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    heatmap = create_heatmap(fixations, size)
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.invert_yaxis()
    return ax


def gaussian(size, sigma):
    """
    Generate a Gaussian kernel.

    Parameters
    ----------
    size : int
        Size of the kernel.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        Normalized Gaussian kernel.
    """
    x = np.linspace(-size // 2 + 1, size // 2, size)
    y = np.linspace(-size // 2 + 1, size // 2, size)
    x, y = np.meshgrid(x, y)
    gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return gauss / gauss.max()


def visualize_data(fixations, size, imagefile=None, savefile=None):
    """
    Visualize fixation data using circles and scanpath overlay.

    Parameters
    ----------
    fixations : dict
        A dictionary with fixation data.
    size : tuple of int
        Width and height of the display in pixels (width, height).
    imagefile : str, optional
        Path to the image file to overlay.
    savefile : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = draw_display(size, imagefile=imagefile)

    # Draw fixations
    draw_circles(
        ax,
        fixations["x_mean"],
        fixations["y_mean"],
        fixations["duration"] / 30.0,
        COLS["chameleon"][2],
        alpha=0.5,
    )

    # Draw scanpath
    for i, (x, y) in enumerate(zip(fixations["x_mean"], fixations["y_mean"]), start=1):
        ax.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha="center")

    for (x1, y1), (x2, y2) in zip(
        zip(fixations["x_mean"], fixations["y_mean"]),
        zip(fixations["x_mean"][1:], fixations["y_mean"][1:]),
    ):
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=3, head_length=5, fc="k", ec="k")

    ax.invert_yaxis()

    if savefile:
        fig.savefig(savefile)

    return fig
