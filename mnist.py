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

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = x**2 - 2*x + 1
print("y.datey = " + str(y.data))
y.backward() # dimension of y is 1, so the grad of y is setted 1 automatically.
print("x.grad = " + str(x.grad))

z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True) # gradients of intermediate variables are released.
print("z.grad = " + str(z.grad))
y.backward()
print("z.grad is None? : " + str((z.grad is None)))

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32) # you should be set y.grad to backward.
y.backward()
print("x.grad = \n" + str(x.grad))

f = L.Linear(3, 2)
print("f.W.data = \n" + str(f.W.data))
print("f.b.data = \n" + str(f.b.data))

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
print("y.data = \n" + str(y.data))

f.cleargrads()
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
print("f.W.grad = \n" + str(f.W.grad)) # df/dw = sigma(x)
print("f.b.grad = \n" + str(f.b.grad)) # df/db = sigma(1)

l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)
def my_forward(x):
    h = l1(x)
    return l2(h)

class MyProc(object):
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l3 = L.Linear(3, 2)
    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)

class MyChain(Chain): # CPU/GPU migration, save/load feature, etc. are supported.
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

class MyChain2(ChainList):
    def __ini__(self):
        super(MyChain2, self).__init__(
                L.Linear(4, 3),
                L.Linear(3, 2),
                )
    def __call__(self, x):
        h = self[0](x)
        return self[1](h)

model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()

def lossfun(arg1, arg2):
    loss = F.sum(model(arg1-arg2))
    return loss

arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))
optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))

serializers.save_npz('my.model', model)
serializers.load_npz('my.model', model)
serializers.save_npz('my.state', optimizer)
serializers.load_npz('my.state', optimizer)

train, test = datasets.get_mnist()
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100,
        repeat=False, shuffle=False) # if repeat == True, infinite loop.

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss' : loss, 'accuracy' : accuracy}, self)
        return loss

model = L.Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
