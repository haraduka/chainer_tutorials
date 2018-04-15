import sys, os
import readline
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


# Basic usage of type check
class ExpAdd(Function):
    def check_type_forward(self, in_types):
        x_type, y_type = in_types
        # you can access .shape, .ndim, .dtype
        utils.type_check.expect(x_type.ndim == 2) # here!
        utils.type_check.expect(x_type.shape[0] > 0) # here!
        utils.type_check.expect(x_type.dtype == np.float32) # here!
        utils.type_check.expect(x_type.dtype.kind == 'f') # here!
        utils.type_check.expect(x_type.shape[1] == y_type.shape[1]) # here!
        #utils.type_check.expect(x_type.shape[0] == y_type.shape[0] * 4)
        #sum = utils.type_check.Variable(np.sum, 'sum')
        #utils.type_check.expect(sum(x_type.shape) == 10)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        z = xp.exp(x) + xp.exp(y)
        return z,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        gz, = grad_outputs

        gx = gz * xy.exp(x)
        gy = gz * xy.exp(y)
        return gx, gy

def expadd(x, y):
    return ExpAdd()(x, y)

x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
w = expadd(x, y)
print("w = \n" + str(w))

