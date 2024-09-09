"""
This file contains the activation functions for the network
"""
import math
from typing import Any


def sigmoid(x: float) -> float:
    return 1 / 1 + math.e**(-x)


def inverse(x: float) -> float:
    return 1 / x


def absolute(x: float) -> float:
    return abs(x)


def clamped(x: float) -> float:
    # DEFINITION COPIED
    return max(-1.0, min(1.0, x))


def cubed(x: float) -> float:
    return x ** 3


def squared(x: float) -> float:
    return x ** 2


def exponential(x: float) -> float:
    return math.e ** x


def hat(x: float) -> float:
    # DEFINITION COPIED
    return max(0.0, 1 - abs(x))


def gaussian(x: float) -> float:
    return math.e**(-4*x**2)


def log(x: float) -> float:
    return math.log(x)


def sin(x: float) -> float:
    return math.sin(x)


def relu(x: float) -> float:
    return x if x > 0 else 0


def leaky_relu(x: float) -> float:
    leak = 1e-2
    return x if x > 0 else x * leak


def softplus(x: float) -> float:
    return math.log(1 + math.e ** x)


def elu(x: float) -> float:
    return math.e**x - 1 if x < 0 else x


def gelu(x: float) -> float:
    return 0.5 * x * (1 + math.erf(x / math.sqrt(x)))


def scaled_elu(x: float) -> float:
    alpha_scale = 1.847
    scale = 1.000002
    return scale*alpha_scale*(math.e**x - 1) if x < 0 else scale*x


def tanh(x: float) -> float:
    return math.tanh(x)


class ActivationFunctions:
    functions: dict[str, Any] = {
        "sigmoid": sigmoid,
        "inv": inverse,
        "abs": absolute,
        "clamped": clamped,
        "squared": squared,
        "cubed": cubed,
        "exp": exponential,
        "hat": hat,
        "gauss": gaussian,
        "log": log,
        "sin": sin,
        "relu": relu,
        "lrelu": leaky_relu,
        "softplus": softplus,
        "elu": elu,
        "selu": scaled_elu,
        "gelu": gelu,
        "tanh": tanh,
        "None": None
    }

    enabled_functions: tuple = tuple(functions.keys())

    def enable_functions(self, functions: tuple) -> None:
        self.enabled_functions = ()

        for function in functions:
            if function in self.functions.keys:
                self.enabled_functions = self.enabled_functions + (function, )
            else:
                raise KeyError("invalid function")


Activations = ActivationFunctions()

"""
the function bases where taken from: https://neat-python.readthedocs.io/en/latest/activation.html,
                                     https://www.theaidream.com/post/an-overview-of-activation-functions-in-deep-learning

most of the exact function definitions weren't copied except for a few functions for performance reasons.
"""