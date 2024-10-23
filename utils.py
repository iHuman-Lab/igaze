import sys
from contextlib import contextmanager, suppress


class SkipWithError(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ["skip"]
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip("{:>12}  {:>2}  {:>12}".format("Skipping the block", "|", f))
            raise SkipWithError
        else:
            p.print_run("{:>12}  {:>3}  {:>12}".format("Running the block", "|", f))
            yield

    with suppress(SkipWithError):
        yield check_active


class ColorPrint:
    @staticmethod
    def print_skip(message, end="\n"):
        sys.stderr.write("\x1b[88m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_run(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_warn(message, end="\n"):
        sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)
