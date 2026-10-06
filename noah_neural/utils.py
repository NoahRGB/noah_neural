import numpy as np

def unbroadcast(g, shape):
    # if numpy has broadcasted the tensor to perform
    # the forward pass, it needs to be "unbroadcasted"
    # to match the original shape

    # this is achieved by summing the computed gradient
    # until it matches the original shape of the tensor

    # TODO understand this more

    while len(g.shape) > len(shape):
        g = g.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            g = g.sum(axis=i, keepdims=True)
    return g
