"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

# Implementation of a prelude of elementary functions.

# Mathematical functions:


# - mul
def mul(x: float, y: float) -> float:
    return x * y


# - id
def id(x: float) -> float:
    return x


# - add
def add(x: float, y: float) -> float:
    return x + y


# - neg
def neg(x: float) -> float:
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    return x == y


# - max
def max(x: float, y: float) -> float:
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


# - sigmoid
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    return max(0.0, x)


#  - log
def log(x: float) -> float:
    return math.log(x)


# - exp
def exp(x: float) -> float:
    return math.exp(x)


# - log_back
def log_back(x: float, d: float) -> float:
    return d / x


# - inv
def inv(x: float) -> float:
    return 1.0 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


# - relu_back
def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


# - log_back
def log_back(x: float, d: float) -> float:
    return d / x


# - exp
def exp_back(x: float, d: float) -> float:
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
    return [f(x) for x in iter]


def zipWith(
    f: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    return [f(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    f: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    return functools.reduce(f, iter, init)


def negList(iter: Iterable[float]) -> Iterable[float]:
    return map(neg, iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    return reduce(add, iter, 0.0)


def prod(iter: Iterable[float]) -> float:
    return reduce(mul, iter, 1.0)
