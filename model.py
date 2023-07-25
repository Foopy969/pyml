import numpy as np
from alive_progress import alive_bar

class Model:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def __repr__(self):
        return '\n'.join(str(i) for i in self.layers) + '\n' + str(self.optimizer)
    
    def forward(self, x):
        a = [x] 
        for i in self.layers:
            a[-1] = i.activate(a[-1])
            a.append(i.forward(a[-1]))
        return a

    def backward(self, a, y):
        d = [a[-1] - y]
        for i in np.flip(self.layers):
            d.insert(0, i.backward(d[0]))
            d[0] *= i.deactivate(a[-len(d)])
        return d
    
    def get_gradient(self, x, y):
        states = self.forward(x)
        errors = self.backward(states, y)
        gradient = np.empty((len(self.layers), 2), dtype=np.ndarray)
        for i in range(len(self.layers)):
            gradient[i] = [np.outer(states[i], errors[i+1]), errors[i+1]]
        return gradient
    
    def cost(self, guess, label):
        return np.sum(np.square(guess - label))

    def update(self, gradient):
        for i, j in zip(self.layers, gradient):
            i.update(j)

    def fit(self, x_tr, x_val, y_tr, y_val, epoch=10):
        for i in range(epoch):
            print(f'\nepoch: {i} / {epoch - 1}')
            with alive_bar(len(x_tr)) as bar:
                for x, y in zip(x_tr, y_tr):
                    self.optimizer.train(self, x, y)
                    bar()

            self.eval(x_val, y_val)

    def eval(self, x_val, y_val):
        correct = 0
        loss = 0

        for x, y in zip(x_val, y_val):
            guess = self.forward(x)[-1]
            loss += self.cost(guess, y)
        
            if np.argmax(guess) == np.argmax(y):
                correct += 1

        print('avg loss:', loss / len(x_val))  
        print('correct:', correct, '/', len(x_val)) 
