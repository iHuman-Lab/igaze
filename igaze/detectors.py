# -*- coding: utf-8 -*-
"""PyGazeAnalyser is a Python module for easily analysing eye-tracking data.

Copyright (C) 2014  Edwin S. Dalmaijer.
"""

import numpy

from igaze.gazeplotter import parse_fixations


def blink_detection(x, y, time, missing=0.0, minlen=10):
    """Detects blinks, defined as a period of missing data that lasts for at
    least a minimal amount of samples arguments.

    Parameters
    ----------
    x : np.array
        numpy array of x positions
    y : np.array
        numpy array of y positions
    time : np.array
        numpy array of timestamps
    missing : float, optional
        value to be used for missing data, by default 0.0
    minlen : int, optional
        integer indicating the minimal amount of consecutive
                missing samples, by default 10

    Returns
    -------
    start_blink: list[list]
        list of lists, each containing [starttime]
    end_blink: list[list]
        list of lists, each containing [starttime, endtime, duration]
    """
    # empty list to contain data
    start_blink = []
    end_blink = []

    # check where the missing samples are
    mx = numpy.array(x == missing, dtype=int)
    my = numpy.array(y == missing, dtype=int)
    miss = numpy.array((mx + my) == 2, dtype=int)

    # check where the starts and ends are (+1 to counteract shift to left)
    diff = numpy.diff(miss)
    starts = numpy.where(diff == 1)[0] + 1
    ends = numpy.where(diff == -1)[0] + 1

    # compile blink starts and ends
    for i in range(len(starts)):
        # get starting index
        s = starts[i]
        # get ending index
        if i < len(ends):
            e = ends[i]
        elif len(ends) > 0:
            e = ends[-1]
        else:
            e = -1
        # append only if the duration in samples is equal to or greater than
        # the minimal duration
        if e - s >= minlen:
            # add starting time
            start_blink.append([time[s]])
            # add ending time
            end_blink.append([time[s], time[e], time[e] - time[s]])

    return start_blink, end_blink


def remove_missing(x, y, time, missing):
    mx = numpy.array(x == missing, dtype=int)
    my = numpy.array(y == missing, dtype=int)
    x = x[(mx + my) != 2]
    y = y[(mx + my) != 2]
    time = time[(mx + my) != 2]
    return x, y, time


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):  # noqa: PLR0913
    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data).

    Parameters
    ----------
    x : np.array
        numpy array of x positions
    y : np.array
        numpy array of y positions
    time : np.array
        numpy array of timestamps
    missing : float, optional
        value to be used for missing data, by default 0.0
    maxdist : int, optional
        maximal inter sample distance in pixels, by default 25
    mindur : int, optional
        minimal duration of a fixation in milliseconds; detected
        fixation cadidates will be disregarded if they are below
        this duration, by default 50

    Returns
    -------
    start_fixation: list[lists]
        list of lists, each containing [starttime]
    end_fixation: list[list]
        list of lists, each containing [starttime, endtime, duration, endx, endy]
    """

    x, y, time = remove_missing(x, y, time, missing)

    # empty list to contain data
    start_fixation = []
    end_fixation = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate

        squared_distance = (x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2
        dist = 0.0
        if squared_distance > 0:
            dist = squared_distance**0.5

        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            start_fixation.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - start_fixation[-1][0] >= mindur:
                end_fixation.append(
                    [start_fixation[-1][0], time[i - 1], time[i - 1] - start_fixation[-1][0], x[si], y[si]],
                )
            # delete the last fixation start if it was too short
            else:
                start_fixation.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1
    # Add last fixation end (we can lose it if dist > maxdist
    # is false for the last point)
    if len(start_fixation) > len(end_fixation):
        end_fixation.append(
            [start_fixation[-1][0], time[len(x) - 1], time[len(x) - 1] - start_fixation[-1][0], x[si], y[si]]
        )
    return start_fixation, end_fixation


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):  # noqa: PLR0913
    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity of over a velocity threshold or an acceleration threshold.

    Parameters
    ----------
    x : np.array
        numpy array of x positions
    y : np.array
        numpy array of y positions
    time : np.array
        numpy array of timestamps
    missing : float, optional
        value to be used for missing data, by default 0.0
    minlen : int, optional
        minimal length of saccades in milliseconds; all detected
        saccades with len(sac) < minlen will be ignored, by default 5
    maxvel : int, optional
        velocity threshold in pixels/second, by default 40
    maxacc : int, optional
        acceleration threshold in pixels, by default 340

    Returns
    -------
    start_saccade : list[list]
        list of lists, each containing [starttime]
    end_saccade	: list[list]
        list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    """

    x, y, time = remove_missing(x, y, time, missing)

    # CONTAINERS
    start_saccade = []
    end_saccade = []

    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = (numpy.diff(x) ** 2 + numpy.diff(y) ** 2) ** 0.5
    # get inter-sample times
    inttime = numpy.diff(time)
    # recalculate inter-sample times to seconds
    inttime = inttime / 1000.0

    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = numpy.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold

        # detect saccade starts
        sacstarts = numpy.where((vel[1 + t0i :] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time) - 1:
                t1i = len(time) - 2
            t1 = time[t1i]

            # add to saccade starts
            start_saccade.append([t1])

            # detect saccade endings
            sacends = numpy.where((vel[1 + t1i :] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time) - 1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    end_saccade.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    start_saccade.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True

    return start_saccade, end_saccade


def scan_path(fixations):
    """Length of the scan path

    Parameters
    ----------
    fixations : _type_
        _description_

    Returns
    -------
    float
        scan path
    """
    fix = parse_fixations(fixations)
    path = numpy.array([fix["x"], fix["y"]]).transpose()
    path_length = numpy.linalg.norm(numpy.diff(path, axis=0), axis=1)
    return numpy.nanmean(path_length)
