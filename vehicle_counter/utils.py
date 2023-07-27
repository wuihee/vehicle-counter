import os
from collections import namedtuple


class Colors:
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)


Point = namedtuple("Point", ["x", "y"])


def change_to_cwd() -> None:
    """
    Change directory to the directory of the current file.
    """
    os.chdir(os.path.realpath(os.path.dirname(__file__)))
