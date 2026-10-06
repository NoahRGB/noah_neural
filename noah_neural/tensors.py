import numpy as np

import noah_neural.operators as ops

class Tensor:

    def __init__(self, shape: tuple = None, data: np.ndarray = None, op=None, inputs=tuple()):
        if shape is None:
            if data is None:
                raise ValueError("Must provide shape or data to Tensor")
            self.shape = data.shape
            self.val = data
        else:
            self.shape = shape
            self.val = np.zeros(shape)

        self.op = op if op is not None else ops.Noop()
        self.inputs = inputs
        self.grad = np.zeros(self.shape) 
    
    def get_op(self):
        return self.op

    def get_inputs(self):
        return self.inputs

    def bprop(self):
        
        # make sure that all nodes before this one
        # have had their backprop completed
        # from kaparthy series
        order = []
        visited = set()
        def get_topological_order(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for input in tensor.inputs:
                    get_topological_order(input)
                order.append(tensor)

        get_topological_order(self)

        # zero grads out
        for tensor in reversed(order):
            tensor.grad = np.zeros(tensor.shape)

        self.grad = np.ones(self.shape)
        for tensor in reversed(order):
            tensor.op.bprop()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=np.array(other))
        op = ops.AddOp(self, other)
        return op.result

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=np.array(other))
        return self + (-other)

    def __mul__(self, other):
        # elementwise
        other = other if isinstance(other, Tensor) else Tensor(data=np.array(other))
        op = ops.MulOp(self, other)
        return op.result

    def __matmul__(self, other):
        assert isinstance(other, Tensor)
        op = ops.MatMulOp(self, other)
        return op.result

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        op = ops.PowOp(self, other)
        return op.result

    def sum(self, axis=None, keepdims=False):
        op = ops.SumOp(self, axis, keepdims)
        return op.result

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.val.size
        else:
            n = self.shape[axis] 
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        op = ops.ReluOp(self)
        return op.result

    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * other**-1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __str__(self):
        return f"Tensor: val={self.val}"
