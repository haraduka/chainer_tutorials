import sys, os, math
import readline
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

## skelton
#class Hoge(Function):
#    def forward_cpu(self, inputs):
#        # do forward computation on CPU
#        return some_tuple
#    def backward_cpu(self, inputs, grad_outputs):
#        # do backward computation on CPU
#        return some_tuple

class MulAdd(Function):
    def forward_cpu(self, inputs):
        x, y, z = inputs
        w = x*y+z
        return w,
    def backward_cpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y*gw
        gy = x*gw
        gz = gw
        return gx, gy, gz
    # if forward/backward_cpu/gpu are same, you can reduce _cpu/gpu, so only forward or backward.
    def forward_gpu(self, inputs):
        x, y, z = inputs
        w = x*y+z
        return w,
    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y*gw
        gy = x*gw
        gz = gw
        return gx, gy, gz

def mulladd(x, y, z):
    return MulAdd()(x, y, z)

x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
w = mulladd(x, y, z)
print("w = \n" + str(w))

class ExpAdd(Function):
    def forward_cpu(self, inputs):
        x, y = inputs
        z = np.exp(x) + np.exp(y)
        return z,
    def backward_cpu(self, inputs, grad_outputs):
        x, y = inputs
        gz, = grad_outputs

        gx = gz * np.exp(x)
        gy = gz * np.exp(y)
        return gx, gy
    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, y = inputs
        z = cupy.exp(x) + cupy.exp(y)
        return z,
    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, y = inputs
        gz, = grad_outputs

        gx = gz * cupy.exp(x)
        gy = gz * cupy.exp(y)
        return gx, gy

class ExpAdd(Function):
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

w = expadd(x, y)
print("w = \n" + str(w))


# Write an Elementwise Kernel Function
# This can be only used for float32, so we can rewrite float32 to T (this means arbitary).
class MulAdd(Function):
    def forward_cpu(self, inputs):
        x, y, z = inputs
        w = x*y+z
        return w,
    def backward_cpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y*gw
        gy = x*gw
        gz = gw
        return gx, gy, gz
    # if forward/backward_cpu/gpu are same, you can reduce _cpu/gpu, so only forward or backward.
    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, y, z = inputs
        w = cuda.elementwise(
                'float32 x, float32 y, float32 z',
                'float32 w'
                'w = x * y + z',
                'muladd_fwd')(x, y, z)
        return w,
    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx, gy = cuda.elementwise(
                'float32 x, float32 y, float32 gw',
                'float32 gx, float32 gy',
                '''
                    gx = y * gw;
                    gy = x * gw;
                '''
                'muladd_bwd')(x, y, gw)
        gz = gw
        return gx, gy, gz

def mulladd(x, y, z):
    return MulAdd()(x, y, z)

w = mulladd(x, y, z)
print("w = \n" + str(w))


# Write a function with training/test mode
def dropout(x):
    xp = cuda.get_array_module(x.data)
    mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
    return x * mask

def dropout(x):
    if not chainer.config.train:
        return x

    xp = cuda.get_array_module(x.data)
    mask = 2 * (xp.random.rand(*x.shape) > 0.5).astype(x.dtype)
    return x * mask

print("dropout = \n" + str(dropout(x)))


# Links that wrap functions
class EltwiseParamProduct(Link):
    def __init__(self, shape):
        super(EltwiseParamProduct, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(initializers.Normal(scale=1.), shape)
    def __call__(self, x):
        return self.W * x

class LinearFunction(Function):
    def forward(self, inputs):
        x, W, b = inputs
        return x.dot(W.T) + b,
    def backward(self, inputs, grad_outputs):
        x, W, b = inputs
        gy, = grad_outputs

        gx = gy.dot(W)
        gW = gy.T.dot(x)
        gb = gy.sum(axis=0)
        return gx, gW, gb

def linear(x, W, b):
    return LinearFunction()(x, W, b)

class Linear(Link):
    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(
                    initializers.Normal(1./math.sqrt(in_size)),
                    (out_size, in_size))
            self.b = chainer.Parameter(0, (out_size,))
    def __call__(self, x):
        return linear(x, self.W, self.b)

print("W = \n" + str(Linear(3, 2).W.data))
print("b = \n" + str(Linear(3, 2).b.data))


# Testing Function
# calcurate grad numerically and compare with the output of backward!
x = np.random.randn(4, 3).astype(np.float32)
gy = np.ones((4, 3), dtype=np.float32)
f = lambda: (x *x,)
gx = gradient_check.numerical_grad(f, (x,), (gy,))
print("x = \n" + str(x))
print("gx = \n" + str(gx))

import unittest
from chainer import testing

class TestReLU(unittest.TestCase):
    def test_backward_cpu(self):
        x = Variable(np.random.randn(3, 2).astype(np.float32))
        y = F.relu(x)
        y.grad = np.random.randn(3, 2).astype(np.float32)
        y.backward()

        def f():
            return F.relu(x).data,

        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad, ))
        testing.assert_allclose(gx, x.grad)

class TestReLU2(unittest.TestCase):
    def test_backward_cpu(self):

        def f(x):
            return F.relu(x)

        x = np.random.randn(3, 2).astype(np.float32)
        y_grad = np.random.randn(3, 2).astype(np.float32)
        gradient_check.check_backward(f, x, y_grad, atol=1e-4, rtol=1e-4)

unittest.main()
