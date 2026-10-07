import numpy as np

import noah_neural.tensors as tensors

class ReLU:
    def __call__(self, x):
        return x.relu()

    def gradient_descent(self, lr):
        ...

class Network:

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, inp):

        out = inp
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out

    def params(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params.extend(layer.params())
        return params

    def gradient_descent(self, lr):
        for layer in self.layers:
            layer.gradient_descent(lr)

    def __call__(self, inp):
        return self.forward(inp)


class LinearLayer:

    def __init__(self, in_size: int, out_size: int):
        self.in_size = in_size; self.out_size = out_size

        # initialised weights how PyTorch do theirs
        k = 1 / in_size
        self.weights = tensors.Tensor(
            data=np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(in_size, out_size)) 
        )

        self.biases = tensors.Tensor(shape=(out_size,))

    def forward(self, inp):
        out = inp @ self.weights + self.biases
        return out

    def gradient_descent(self, lr):
        for param in self.params():
            param.val -= lr * param.grad

    def params(self):
        return [self.weights, self.biases]

    def __call__(self, inp):
        return self.forward(inp)

