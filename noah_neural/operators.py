import numpy as np

from noah_neural.utils import unbroadcast
import noah_neural.tensors as tensors

class Noop:
    def bprop(self):
        return None 

class AddOp:
    def __init__(self, a, b):
        self.a = a; self.b = b
        self.result = tensors.Tensor(data=a.val+b.val, op=self, inputs=(self.a, self.b))

    def bprop(self):
        self.a.grad += unbroadcast(self.result.grad, self.a.shape)
        self.b.grad += unbroadcast(self.result.grad, self.b.shape)

class MulOp:
    # elementwise
    def __init__(self, a, b):
        self.a = a; self.b = b
        self.result = tensors.Tensor(data=a.val*b.val, op=self, inputs=(self.a, self.b))

    def bprop(self):
        self.a.grad += unbroadcast(self.b.val * self.result.grad, self.a.shape)
        self.b.grad += unbroadcast(self.a.val * self.result.grad, self.b.shape)

class MatMulOp:
    def __init__(self, a, b):
        self.a = a; self.b = b
        self.result = tensors.Tensor(data=a.val@b.val, op=self, inputs=(self.a, self.b))

    def bprop(self):
        self.a.grad += self.result.grad @ self.b.val.T
        self.b.grad += self.a.val.T @ self.result.grad

class ExpOp:
    def __init__(self, a):
        self.a = a
        self.result = tensors.Tensor(data=np.exp(self.a), op=self, inputs=(self.a,))

    def bprop(self):
        self.a.grad += self.result.val * self.result.grad

class ReluOp:
    def __init__(self, a):
        self.a = a
        self.result = tensors.Tensor(data=np.maximum(0, self.a.val), op=self, inputs=(self.a,))

    def bprop(self):
        # only let grads flow through when val > 0
        # since relu is max(0, val)
        self.a.grad += (self.a.val > 0) * self.result.grad

class GetItemOp:
    # used for indexing into a tensor
    def __init__(self, a, indices):
        self.a = a; self.indices = indices
        self.result = tensors.Tensor(data=a.val[indices], op=self, inputs=(self.a,))

    def bprop(self):
        # only let gradients flow through for the "chosen" indices
        grads = np.zeros_like(self.a.val)
        np.add.at(grads, self.indices, self.result.grad)
        self.a.grad += grads

class SumOp:
    def __init__(self, a, axis=None, keepdims=False):
        self.a = a; self.axis=axis; self.keepdims = keepdims
        self.result = tensors.Tensor(data=np.sum(a.val, axis=axis, keepdims=keepdims), op=self, inputs=(self.a,))

    def bprop(self):
        grad = self.result.grad
        if not self.keepdims and self.axis is not None:
           grad = np.expand_dims(grad, self.axis)
        self.a.grad += np.broadcast_to(grad, self.a.shape)

class PowOp:
    def __init__(self, a, b):
        assert isinstance(b, (int,float))
        self.a = a; self.b = b
        self.result = tensors.Tensor(data=a.val**b, op=self, inputs=(self.a,))

    def bprop(self):
        self.a.grad += self.b * (self.a.val ** (self.b-1)) * self.result.grad

