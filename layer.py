from abc import ABC, abstractmethod
import numpy as np

def one_hot(n, i):
    v = np.zeros(n)
    v[i] = 1
    return v

class Base(ABC):
    def set_activation(self, activation):
        match(activation):
            case 'SIGMOID':
                self.activate = lambda x: 1 / (1 + np.exp(-x))
                self.deactivate = lambda x: x * (1 - x)
            case 'RELU':
                self.activate = lambda x: np.maximum(0, x)
                self.deactivate = lambda x: np.where(x > 0, 1, 0)
            case 'TANH':
                self.activate = lambda x: np.tanh(x)
                self.deactivate = lambda x: 1.0 - np.tanh(x)**2
            case 'NONE':
                self.activate = lambda x: x
                self.deactivate = lambda x: x
        self.activation = activation

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, input):
        pass

    def process(self, input):
        return input
    
    def activate(self):
        pass

    def deactivate(self):
        pass

class Dense(Base):
    def __init__(self, w, h, activation):
        self.w = np.random.rand(w, h)
        self.b = np.random.rand(h)

        self.set_activation(activation)

    def __repr__(self):
        return f"Dense(({self.w.shape}), {self.activation})"

    def forward(self, a):
        return np.dot(a, self.w) + self.b

    def backward(self, d):
        return np.dot(d, self.w.T)
    
    def update(self, gradient):
        self.w -= gradient[0]
        self.b -= gradient[1]
    
class Embedding(Dense):
    def __init__(self, w, words):
        super().__init__(len(words), w, 'NONE')
        self.words = words

        self.activate = lambda x: one_hot(len(self.words), self.words.index(x))

    def __repr__(self):
        return f"Embedding(({self.w.shape}))"

