from typing import Callable, Dict
import numpy as np
from numpy.typing import NDArray

class CoefficientFunction:
    def __call__(x:NDArray[np.floating], y:NDArray[np.floating], region_index:int):
        raise NotImplemented("Not implemented")

class ConstantCoefficientFunction:
    def __init__(self, value: float):
        self.value = value
    def __call__(self, x:NDArray[np.floating], y:NDArray[np.floating], region_index:int=0):
        return self.value * np.ones_like(x)

class DomainConstantCoefficientFunction:
    def __init__(self, values: Dict[int, float]):
        self.values = values
    def __call__(self, x:NDArray[np.floating], y:NDArray[np.floating], region_index:int):
        return self.values[region_index] * np.ones_like(x)
    
class VariableCoefficientFunction:
    def __init__(self, functions: Dict[int, Callable[[NDArray[np.floating], NDArray[np.floating]],NDArray[np.floating]]]):
        self.functions = functions
    def __call__(self, x:NDArray[np.floating], y:NDArray[np.floating], region_index:int):
        return self.functions[region_index](x,y)