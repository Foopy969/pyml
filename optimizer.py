from abc import ABC, abstractmethod
import numpy as np

class optimizer(ABC):
    def __init__(self, type):
        self.type = type

    def __repr__(self):
        return f"Optimizer({self.type})"

    def sqrt(self, n):
        return np.frompyfunc(np.sqrt, 1, 1)(n)

    @abstractmethod
    def train(self, model, x_tr, y_tr):
        pass

class SGD(optimizer):
    def __init__(self, rate):
        self.rate = rate
        super().__init__('SGD')

    def train(self, model, x, y):
        gradient = model.get_gradient(x, y)
        model.update(gradient * self.rate)

class BatchSGD(optimizer):
    def __init__(self, size, rate):
        self.size = size
        self.rate = rate
        self.i = 0
        self.g = 0
        super().__init__('BatchSGD')

    def train(self, model, x, y):
        if self.i % self.size == self.size - 1:
            model.update(self.g * self.rate / self.size)
            self.g = 0
        self.g += model.get_gradient(x, y)
        self.i += 1
           

class Adagrad(optimizer):
    def __init__(self, rate, eps=1e-8):
        self.rate = rate
        self.eps = eps
        self.g = 0
        super().__init__('Adagrad')

    def train(self, model, x, y):
        gradient = model.get_gradient(x, y)
        self.g += gradient ** 2
        model.update(gradient * self.rate / self.sqrt(self.g + self.eps))

class RMSprop(optimizer):
    def __init__(self, rate, rho=0.9, eps=1e-8):
        self.rate = rate
        self.rho = rho
        self.eps = eps
        self.g = 0
        super().__init__('RMSprop')

    def train(self, model, x, y):
        gradient = model.get_gradient(x, y)
        self.g = self.rho * self.g + (1 - self.rho) * gradient ** 2
        model.update(gradient * self.rate / self.sqrt(self.g + self.eps))

class Adam(optimizer):
    def __init__(self, rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.rate = rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = 0
        self.v = 0
        super().__init__('Adam')

    def train(self, model, x, y):
        self.t += 1
        gradient = model.get_gradient(x, y)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        model.update(m_hat * self.rate / self.sqrt(v_hat + self.eps))



    

