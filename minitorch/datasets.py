import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N):
    """Generate N random 2D points.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples representing the (x1, x2) coordinates of the points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N):
    """Generates a simple 2D dataset where points are classified based on whether
    their x1 coordinate is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N):
    """Generates a 2D dataset where points are classified based on whether the sum
    of their x1 and x2 coordinates is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N):
    """Generates a 2D dataset where points are classified based on whether
    their x1 coordinate is less than 0.2 or greater than 0.8.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N):
    """Generates a 2D XOR dataset where points are classified based on whether
    one coordinate is less than 0.5 and the other is greater than 0.5.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N):
    """Generates a 2D circular dataset where points are classified based on
    their distance from the center (0.5, 0.5).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N):
    """Generates a 2D spiral dataset where points are arranged in two spirals
    with different labels.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph object containing N, X (list of points), and y (list of labels).

    """

    def x(t):
        return t * math.cos(t) / 20.0

    def y(t):
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
