import numpy as np 
import activations

class Layer:
    """
    | Creates a neural layer.
    | **Example**
    ```
    m_input = [1, 2, 5]
    OUTPUT_SIZE = 5

    net = Sequential(m_input, [
        Layer(32, 3, activation=activations.relu),
        Layer(64, 32, activation=activations.relu),
        Layer(OUTPUT_SIZE, 64, activation=activations.softmax)
    ])
    ```
    """
    
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
    
class Sequential:
    def __init__(self, m_input:list[float], layers:list[Layer]):
        self.m_input = np.array(m_input)
        self.layers = layers
        
    def feedforward(self):
        if(len(self.layers) == 0):
            raise Exception("You must have at least one layer.")
        first = self.layers[0]
        out = first(self.m_input)
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            out = layer(out)
        return out                 
                 
                 
         

# m_input = [1, 2, 5]
# OUTPUT_SIZE = 5

# net = Sequential(m_input, [
#     Layer(32, 3, activation=activations.relu),
#     Layer(64, 32, activation=activations.relu),
#     Layer(OUTPUT_SIZE, 64, activation=activations.softmax)
# ])

# print(net.feedforward())