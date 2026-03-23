import numpy


def remove_missing(x, y, time, missing):
    mx = numpy.array(x == missing, dtype=int)
    my = numpy.array(y == missing, dtype=int)
    mask = (mx + my) != 2
    return x[mask], y[mask], time[mask]


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=100, maxdur=700, maxgap=150):
    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data).

    arguments
    x       - numpy array of x positions
    y       - numpy array of y positions
    time    - numpy array of timestamps in milliseconds

    keyword arguments
    missing - value to be used for missing data (default = 0.0)
    maxdist - maximal inter-sample distance in pixels (default = 25)
    mindur  - minimal fixation duration in ms (default = 100)
    maxdur  - maximal fixation duration in ms; longer fixations are removed
              as tracking artifacts (default = 700)
    maxgap  - maximal allowed time gap between consecutive valid samples in ms;
              larger gaps break the fixation (default = 150)

    returns
    Sfix, Efix
        Sfix - list of lists, each containing [starttime]
        Efix - list of lists, each containing [starttime, endtime, duration, endx, endy]
    """

    x, y, time = remove_missing(x, y, time, missing)

    Sfix = []
    Efix = []

    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # break fixation if time gap between samples is too large
        if time[i] - time[i - 1] > maxgap:
            if fixstart:
                dur = time[i - 1] - Sfix[-1][0]
                if mindur <= dur <= maxdur:
                    Efix.append([Sfix[-1][0], time[i - 1], dur, x[si], y[si]])
                else:
                    Sfix.pop(-1)
                fixstart = False
            si = i
            continue

        squared_distance = (x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2
        dist = squared_distance ** 0.5 if squared_distance > 0 else 0.0

        if dist <= maxdist and not fixstart:
            si = i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            fixstart = False
            dur = time[i - 1] - Sfix[-1][0]
            if mindur <= dur <= maxdur:
                Efix.append([Sfix[-1][0], time[i - 1], dur, x[si], y[si]])
            else:
                Sfix.pop(-1)
            si = i
        elif not fixstart:
            si += 1

    # capture last fixation
    if len(Sfix) > len(Efix):
        dur = time[len(x) - 1] - Sfix[-1][0]
        if mindur <= dur <= maxdur:
            Efix.append([Sfix[-1][0], time[len(x) - 1], dur, x[si], y[si]])
        else:
            Sfix.pop(-1)

    return Sfix, Efix
