import numpy as np 
import activations

class Layer:
    def __init__(self, units:int, input_size:int, activation=None, label = None):
        self.label = label 
        self.W = 2 * np.random.rand(units, input_size) - 1
        self.b = 2 * np.random.rand(units) - 1
        self.activation = activation
    
    def __call__(self, x:list[float]):
        _x = np.array(x)
        ret = np.add(np.matmul(self.W, _x), self.b)
        if self.activation:
            return self.activation(ret)
        return ret 

m_input = [1, 2, 5]
x1 = Layer(32, 3, activation=activations.relu)(m_input)
x2 = Layer(64, 32, activation=activations.relu)(x1)
output = Layer(5, 64, activation=activations.softmax)(x2)
print(output)