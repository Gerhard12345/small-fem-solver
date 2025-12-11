from typing import Callable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray


class CoefficientFunction:
    def __call__(self, x: NDArray[np.floating], y: NDArray[np.floating], region_index: int) -> NDArray[np.floating]:
        raise NotImplementedError("Not implemented")


class ConstantCoefficientFunction(CoefficientFunction):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x: NDArray[np.floating], y: NDArray[np.floating], region_index: int = 0) -> NDArray[np.floating]:
        return np.matlib.repmat(self.value, *x.shape)


class DomainConstantCoefficientFunction(CoefficientFunction):
    def __init__(self, values: Dict[int, float]):
        self.values = values

    def __call__(self, x: NDArray[np.floating], y: NDArray[np.floating], region_index: int) -> NDArray[np.floating]:
        return np.matlib.repmat(self.values[region_index], *x.shape)


class VariableCoefficientFunction(CoefficientFunction):
    def __init__(self, functions: Dict[int, Callable[[float, float], NDArray[np.floating] | float]], f_shape: Tuple[int, int]):
        self.functions = functions
        self.f_shape = f_shape

    def __call__(self, x: NDArray[np.floating], y: NDArray[np.floating], region_index: int) -> NDArray[np.floating]:
        values = np.array([self.functions[region_index](xi, yi) for xi, yi in zip(x.reshape(x.size), y.reshape(y.size))])  # type:ignore
        return values.reshape(x.size * self.f_shape[0], self.f_shape[1])
