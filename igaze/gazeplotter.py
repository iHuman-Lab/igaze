# -*- coding: utf-8 -*-
"""
This file is part of PyGaze - the open-source toolbox for eye tracking

PyGazeAnalyser is a Python module for easily analysing eye-tracking data
Copyright (C) 2014  Edwin S. Dalmaijer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
# Gaze Plotter
#
# Produces different kinds of plots that are generally used in eye movement
# research, e.g. heatmaps, scanpaths, and fixation locations as overlays of
# images.
#
# version 2 (02 Jul 2014)

__author__ = "Edwin Dalmaijer"

# native
import os

# external
import numpy
from matplotlib import pyplot
from PIL import Image

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
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


def draw_fixations(  # noqa: PLR0913
    fixations,
    dispsize,
    imagefile=None,
    durationsize=True,
    durationcolour=True,
    alpha=0.5,
    savefilename=None,
):
    """
    Draw circles on fixation locations, optionally overlaying an image,
    with optional weighting of duration for circle size and color.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the heatmap is to be laid,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    durationsize : bool, optional
        If True, the fixation duration is taken into account as a weight
        for the circle size; longer duration = bigger (default is True).

    durationcolour : bool, optional
        If True, the fixation duration is taken into account as a weight
        for the circle color; longer duration = hotter (default is True).

    alpha : float, optional
        A float between 0 and 1, indicating the transparency of the heatmap,
        where 0 is completely transparent and 1 is completely opaque
        (default is 0.5).

    savefilename : str, optional
        Full path to the file in which the heatmap should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the fixations.
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # CIRCLES
    # duration weigths
    siz = 1 * (fix["dur"] / 30.0) if durationsize else 1 * numpy.median(fix["dur"] / 30.0)
    col = fix["dur"] if durationcolour else COLS["chameleon"][2]

    # draw circles
    ax.scatter(fix["x"], fix["y"], s=siz, c=col, marker="o", cmap="jet", alpha=alpha, edgecolors="none")

    # save the figure if a file name was provided
    if savefilename is not None:
        fig.savefig(savefilename)

    return fig


def draw_heatmap(fixations, dispsize, ax, imagefile=None, alpha=0.5, savefilename=None):  # noqa: PLR0913
    """Draw a heatmap of the provided fixations, optionally overlaying an
    image, and optionally weighting fixations with a higher duration.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the heatmap is to be laid,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    durationweight : bool, optional
        If True, the fixation duration is taken into account as a weight
        for the heatmap intensity; longer duration = hotter (default is True).

    alpha : float, optional
        A float between 0 and 1, indicating the transparency of the heatmap,
        where 0 is completely transparent and 1 is completely opaque
        (default is 0.5).

    savefilename : str, optional
        Full path to the file in which the heatmap should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the heatmap.
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh / 6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(len(fix["dur"])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = int(strt) + int(fix["x"][i]) - int(gwh / 2)
        y = int(strt) + int(fix["y"][i]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if x < 0:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if y < 0:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:  # noqa: SIM105
                heatmap[y : y + vadj[1], x : x + hadj[1]] += gaus[vadj[0] : vadj[1], hadj[0] : hadj[1]] * fix["dur"][i]
            except ValueError:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y : y + gwh, x : x + gwh] += gaus * fix["dur"][i]
    # resize heatmap
    heatmap = heatmap[strt : dispsize[1] + strt, strt : dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN

    # draw heatmap on top of image
    ax.imshow(heatmap, cmap="jet", alpha=alpha)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    pyplot.pause(0.001)
    ax.cla()

    # FINISH PLOT
    # save the figure if a file name was provided
    if savefilename is not None:
        fig.savefig(savefilename)


def draw_eye_heatmap(positions, dispsize, ax, alpha=0.5):
    """
    Draw a heatmap of the provided fixations, optionally overlaying an
    image, and optionally weighting fixations with a higher duration.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the heatmap is to be laid,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    durationweight : bool, optional
        If True, the fixation duration is taken into account as a weight
        for the heatmap intensity; longer duration = hotter (default is True).

    alpha : float, optional
        A float between 0 and 1, indicating the transparency of the heatmap,
        where 0 is completely transparent and 1 is completely opaque
        (default is 0.5).

    savefilename : str, optional
        Full path to the file in which the heatmap should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the heatmap.
    """

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh / 6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    # get x and y coordinates
    # x and y - indexes of heatmap array. must be integers
    for position in positions:
        x = int(strt) + int(position[0]) - int(gwh / 2)
        y = int(strt) + int(position[1]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if x < 0:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if y < 0:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:  # noqa: SIM105
                heatmap[y : y + vadj[1], x : x + hadj[1]] += gaus[vadj[0] : vadj[1], hadj[0] : hadj[1]]
            except ValueError:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y : y + gwh, x : x + gwh] += gaus
    # resize heatmap
    heatmap = heatmap[strt : dispsize[1] + strt, strt : dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN

    # draw heatmap on top of image
    ax.imshow(heatmap, cmap="jet", alpha=alpha)


def animate_heatmap(fixations, dispsize, ax, imagefile=None, durationweight=True, alpha=0.5, savefilename=None):
    """
    Draw a heatmap of the provided fixations, optionally overlaying an
    image, and optionally weighting fixations with a higher duration.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the heatmap is to be laid,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    durationweight : bool, optional
        If True, the fixation duration is taken into account as a weight
        for the heatmap intensity; longer duration = hotter (default is True).

    alpha : float, optional
        A float between 0 and 1, indicating the transparency of the heatmap,
        where 0 is completely transparent and 1 is completely opaque
        (default is 0.5).

    savefilename : str, optional
        Full path to the file in which the heatmap should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the heatmap.
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # IMAGE
    # fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh / 6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(fix["dur"])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = int(strt) + int(fix["x"][i]) - int(gwh / 2)
        y = int(strt) + int(fix["y"][i]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if x < 0:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if y < 0:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:  # noqa: SIM105
                heatmap[y : y + vadj[1], x : x + hadj[1]] += gaus[vadj[0] : vadj[1], hadj[0] : hadj[1]] * fix["dur"][i]
            except ValueError:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y : y + gwh, x : x + gwh] += gaus * fix["dur"][i]
    # resize heatmap
    heatmap = heatmap[strt : dispsize[1] + strt, strt : dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN

    # draw heatmap on top of image
    ax.imshow(heatmap, cmap="jet", alpha=alpha)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    pyplot.pause(0.001)
    ax.cla()


def draw_raw(x, y, dispsize, imagefile=None, savefilename=None):
    """
    Draw the raw x and y data.

    Parameters
    ----------
    x : list
        A list of x coordinates of all samples to be plotted.

    y : list
        A list of y coordinates of all samples to be plotted.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the data is to be laid,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    savefilename : str, optional
        Full path to the file in which the plot should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the plotted data.
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # plot raw data points
    ax.plot(x, y, "o", color=COLS["aluminium"][0], markeredgecolor=COLS["aluminium"][5])

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename is not None:
        fig.savefig(savefilename)

    return fig


def draw_scanpath(fixations, saccades, dispsize, imagefile=None, alpha=0.5, savefilename=None):  # noqa: PLR0913
    """
    Draw a scanpath: a series of arrows between numbered fixations,
    optionally overlaying an image.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    saccades : list
        A list of saccade ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Esac']`.

    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file over which the scanpath is to be drawn,
        or None for no image. Note: the image may be smaller than the
        display size, and the function assumes that the image was
        presented at the center of the display (default is None).

    alpha : float, optional
        A float between 0 and 1 indicating the transparency of the scanpath,
        where 0 is completely transparent and 1 is completely opaque
        (default is 0.5).

    savefilename : str, optional
        Full path to the file in which the scanpath should be saved,
        or None to not save the file (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance containing the scanpath.
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # FIXATIONS
    # parse fixations
    fix = parse_fixations(fixations)
    # draw fixations
    ax.scatter(
        fix["x"],
        fix["y"],
        s=(1 * fix["dur"] / 30.0),
        c=COLS["chameleon"][2],
        marker="o",
        cmap="jet",
        alpha=alpha,
        edgecolors="none",
    )
    # draw annotations (fixation numbers)
    for i in range(len(fixations)):
        ax.annotate(
            str(i + 1),
            (fix["x"][i], fix["y"][i]),
            color=COLS["aluminium"][5],
            alpha=1,
            horizontalalignment="center",
            verticalalignment="center",
            multialignment="center",
        )

    # SACCADES
    if saccades:
        # loop through all saccades
        for _, _, _, sx, sy, ex, ey in saccades:
            # draw an arrow between every saccade start and ending
            ax.arrow(
                sx,
                sy,
                ex - sx,
                ey - sy,
                alpha=alpha,
                fc=COLS["aluminium"][0],
                ec=COLS["aluminium"][5],
                fill=True,
                shape="full",
                width=10,
                head_width=20,
                head_starts_at_zero=False,
                overhang=0,
            )

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename is not None:
        fig.savefig(savefilename)

    return fig


def draw_display(dispsize, imagefile=None):
    """
    Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background color, and optionally with an image drawn
    onto it.

    Parameters
    ----------
    dispsize : tuple or list
        A tuple or list indicating the size of the display, e.g. (1024, 768).

    imagefile : str, optional
        Full path to an image file to be drawn onto the figure, or None
        for no image. Note: the image may be smaller than the display size,
        and the function assumes that the image was presented at the center
        of the display (default is None).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        A matplotlib Figure instance with a black background.

    ax : matplotlib.pyplot.Axes
        The axes of the figure, which may contain the image if provided.
    """

    # construct screen (black background)
    _, ext = os.path.splitext(imagefile)
    ext = ext.lower()
    data_type = "float32" if ext == ".png" else "uint8"
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile is not None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        # img = image.imread(imagefile)
        img = Image.open(imagefile)
        img = img.resize((1500, 750))
        img = numpy.flip(numpy.array(img), axis=0)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if not os.name == "nt":
            # img = numpy.flipud(img) # NOTE: Need not do this in mac
            pass
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)

        # draw the image on the screen
        screen[y : y + h, x : x + w, :] += img[:, :, 0:3]
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(img, origin="upper")

    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    """
    Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution.

    Parameters
    ----------
    x : int
        Width in pixels.

    sx : float
        Width standard deviation.

    y : int, optional
        Height in pixels (default is x).

    sy : float, optional
        Height standard deviation (default is sx).

    Returns
    -------
    numpy.ndarray
        A 2D array representing the Gaussian distribution values.
    """

    # square Gaussian if only x values are passed
    if y is None:
        y = x
    if sy is None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy)))
            )

    return M


def parse_fixations(fixations):
    """
    Returns all relevant data from a list of fixation ending events.

    Parameters
    ----------
    fixations : list
        A list of fixation ending events from a single trial, as produced
        by `edfreader.read_edf`, e.g. `edfdata[trialnr]['events']['Efix']`.

    Returns
    -------
    dict
        A dictionary with three keys: 'x', 'y', and 'dur', each containing
        a numpy array for the x coordinates, y coordinates, and duration
        of each fixation, respectively.
    """

    # empty arrays to contain fixation coordinates
    fix = {"x": numpy.zeros(len(fixations)), "y": numpy.zeros(len(fixations)), "dur": numpy.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len(fixations)):
        stime, etime, dur, ex, ey = fixations[fixnr]
        fix["x"][fixnr] = ex
        fix["y"][fixnr] = ey
        fix["dur"][fixnr] = dur

    return fix
