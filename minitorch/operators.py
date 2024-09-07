"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

# Implementation of a prelude of elementary functions.

# Mathematical functions:


# - mul
def mul(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y


# - id
def id(x: float) -> float:
    """Identity function. Returns the input value unchanged.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The input value.

    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Add two floats.

    Args:
    ----
        x (float): First input value.
        y (float): Second input value.

    Returns:
    -------
        float: The sum of the two input values.

    """
    return x + y


# - neg
def neg(x: float) -> float:
    """Negate a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The negated value of the input.

    """
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    """Compare two floats.

    Args:
    ----
        x (float): First input value.
        y (float): Second input value.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Compare two floats.

    Args:
    ----
        x (float): First input value.
        y (float): Second input value.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Return the maximum of two floats.

    Args:
    ----
        x (float): First input value.
        y (float): Second input value.

    Returns:
    -------
        float: The maximum of the two input values.

    """
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Compare two floats.

    Args:
    ----
        x (float): First input value.
        y (float): Second input value.

    Returns:
    -------
        bool: True if x is close to y, False otherwise.

    """
    return abs(x - y) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    """Compute the sigmoid of a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The sigmoid of the input.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# - relu
def relu(x: float) -> float:
    """Compute the ReLU of a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The ReLU of the input.

    """
    return max(0.0, x)


#  - log
def log(x: float) -> float:
    """Compute the natural logarithm of a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The natural logarithm of the input.

    """
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Compute the exponential of a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The exponential of the input.

    """
    return math.exp(x)


# - log_back
def log_back(x: float, d: float) -> float:
    """Compute the derivative of the log function.

    Args:
    ----
        x (float): Input value.
        d (float): Derivative of the output with respect to the input.

    Returns:
    -------
        float: The derivative of the log function.

    """
    return d / x


# - inv
def inv(x: float) -> float:
    """Compute the inverse of a float.

    Args:
    ----
        x (float): Input value.

    Returns:
    -------
        float: The inverse of the input.

    """
    return 1.0 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the inverse function.

    Args:
    ----
        x (float): Input value.
        d (float): Derivative of the output with respect to the input.

    Returns:
    -------
        float: The derivative of the inverse function.

    """
    return -d / (x * x)


# - relu_back
def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU function.

    Args:
    ----
        x (float): Input value.
        d (float): Derivative of the output with respect to the input.

    Returns:
    -------
        float: The derivative of the ReLU function.

    """
    return d if x > 0 else 0.0


# - log_back
def log_back(x: float, d: float) -> float:
    """Compute the derivative of the log function.

    Args:
    ----
        x (float): Input value.
        d (float): Derivative of the output with respect to the input.

    Returns:
    -------
        float: The derivative of the log function.

    """
    return d / x


# - exp
def exp_back(x: float, d: float) -> float:
    """Compute the derivative of the exponential function.

    Args:
    ----
        x (float): Input value.
        d (float): Derivative of the output with respect to the input.

    Returns:
    -------
        float: The derivative of the exponential function.

    """
    return d * math.exp(x)


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
import functools


def map(f: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        f (Callable[[float], float]): The function to apply.
        iter (Iterable[float]): The iterable to map over.

    Returns:
    -------
        Iterable[float]: The result of applying the function to each element of the iterable.

    """
    return [f(x) for x in iter]


def zipWith(
    f: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to corresponding elements of two iterables.

    Args:
    ----
        f (Callable[[float, float], float]): The function to apply.
        iter1 (Iterable[float]): The first iterable.
        iter2 (Iterable[float]): The second iterable.

    Returns:
    -------
        Iterable[float]: The result of applying the function to corresponding elements of the two iterables.

    """
    return [f(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    f: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    """Reduce an iterable to a single value.

    Args:
    ----
        f (Callable[[float, float], float]): The function to apply.
        iter (Iterable[float]): The iterable to reduce.
        init (float): The initial value.

    Returns:
    -------
        float: The result of reducing the iterable to a single value.

    """
    return functools.reduce(f, iter, init)


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negate each element of an iterable.

    Args:
    ----
        iter (Iterable[float]): The iterable to negate.

    Returns:
    -------
        Iterable[float]: The result of negating each element of the iterable.

    """
    return map(neg, iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two iterables.

    Args:
    ----
        iter1 (Iterable[float]): The first iterable.
        iter2 (Iterable[float]): The second iterable.

    Returns:
    -------
        Iterable[float]: The result of adding corresponding elements of the two iterables.

    """
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sum the elements of an iterable.

    Args:
    ----
        iter (Iterable[float]): The iterable to sum.

    Returns:
    -------
        float: The sum of the elements of the iterable.

    """
    return reduce(add, iter, 0.0)


def prod(iter: Iterable[float]) -> float:
    """Multiply the elements of an iterable.

    Args:
    ----
        iter (Iterable[float]): The iterable to multiply.

    Returns:
    -------
        float: The product of the elements of the iterable.

    """
    return reduce(mul, iter, 1.0)
