import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    RELU = "ReLU"
    LEAKY_RELU = "LeakyRelu"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"

def ReLU(x:np.ndarray) -> np.ndarray:
    return np.maximum(0,x)

def LeakyReLU(x:np.ndarray, alpha:float = 0.01) -> np.ndarray:
    return np.where(x>0, x, alpha*x)

def Sigmoid(x:np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

def Tanh(x:np.ndarray) -> np.ndarray:
    return np.tanh(x)

def Softmax(x:np.ndarray) -> np.ndarray:
    expX = np.exp(x-np.max(x))
    return expX/np.sum(expX)

ActivationFunctions = {
    ActivationFunction.RELU: ReLU,
    ActivationFunction.LEAKY_RELU: LeakyReLU,
    ActivationFunction.SIGMOID: Sigmoid,
    ActivationFunction.TANH: Tanh,
    ActivationFunction.SOFTMAX: Softmax,
}