import numpy


def remove_missing(x, y, time, missing):
    mx = numpy.array(x == missing, dtype=int)
    my = numpy.array(y == missing, dtype=int)
    mask = (mx + my) != 2
    return x[mask], y[mask], time[mask]


def saccade_detection(x, y, time, missing=0.0, minlen=10, maxvel=1000, maxgap=150, maxdur=15):
    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity over a velocity threshold.

    arguments
    x       - numpy array of x positions
    y       - numpy array of y positions
    time    - numpy array of timestamps in milliseconds

    keyword arguments
    missing - value to be used for missing data (default = 0.0)
    minlen  - minimal saccade duration in ms (default = 10)
    maxvel  - velocity threshold in pixels/second (default = 1000)
    maxgap  - maximal allowed time gap between consecutive valid samples in ms;
              larger gaps break the saccade (default = 150)
    maxdur  - maximal saccade duration in ms; longer saccades are discarded
              (default = 15)

    returns
    Ssac, Esac
        Ssac - list of lists, each containing [starttime]
        Esac - list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    """
    x, y, time = remove_missing(x, y, time, missing)

    Ssac = []
    Esac = []

    # inter-sample distance, time, velocity, acceleration
    intdist = (numpy.diff(x) ** 2 + numpy.diff(y) ** 2) ** 0.5
    inttime = numpy.diff(time)

    # break velocity calculation where time gap exceeds maxgap
    gap_mask = inttime > maxgap
    inttime_safe = numpy.where(gap_mask, numpy.nan, inttime / 1000.0)

    vel = intdist / inttime_safe

    t0i = 0
    stop = False
    while not stop:
        sacstarts = numpy.where(
            numpy.nan_to_num(vel[1 + t0i:]) > maxvel,
        )[0]
        if len(sacstarts) > 0:
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time) - 1:
                t1i = len(time) - 2
            t1 = time[t1i]
            Ssac.append([t1])

            sacends = numpy.where(
                numpy.nan_to_num(vel[1 + t1i:]) < maxvel,
            )[0]
            if len(sacends) > 0:
                t2i = sacends[0] + 1 + t1i + 1
                if t2i >= len(time):
                    t2i = len(time) - 1
                t2 = time[t2i]
                dur = t2 - t1

                if minlen <= dur <= maxdur:
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    Ssac.pop(-1)

                t0i = t2i
            else:
                stop = True
        else:
            stop = True

    return Ssac, Esac
