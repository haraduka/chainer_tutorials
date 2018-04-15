import chainer
from chainer import cuda
import cupy
import numpy as np
from chainer import training, optimizers, datasets, iterators
from chainer import Variable, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples

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

## very easy way
#
#model = L.Classifier(MLP(1000, 10))
#chainer.cuda.get_device_from_id(0).use()
#
#optimizer = optimizers.Adam()
#optimizer.setup(model)
#
#updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'second': 1})
#trainer = training.Trainer(updater, (20, 'epoch'), out='result')
#trainer.extend(extensions.Evaluator(test_iter, model, device=0))
#trainer.extend(extensions.LogReport())
#trainer.extend(extensions.PrintReport(
#    ['epoch', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())
#trainer.run()

model_0 = L.Classifier(MLP(1000, 10))
model_1 = model_0.copy()
model_0.to_gpu(0)
model_1.to_gpu(1)

optimizer = optimizers.Adam()
optimizer.setup(model_0)

batchsize = 100
x_train, y_train = concat_examples(train)
datasize = len(x_train)
for epoch in range(20):
    print('epoch: %d' % epoch)
    indexes = np.random.permutation(datasize)
    for i in range(0, datasize, batchsize):
        x_batch = x_train[indexes[i:i+batchsize]]
        y_batch = y_train[indexes[i:i+batchsize]]

        x0 = Variable(cuda.to_gpu(x_batch[:batchsize//2], 0))
        t0 = Variable(cuda.to_gpu(y_batch[:batchsize//2], 0))
        x1 = Variable(cuda.to_gpu(x_batch[:batchsize//2], 1))
        t1 = Variable(cuda.to_gpu(y_batch[:batchsize//2], 1))

        loss_0 = model_0(x0, t0)
        loss_1 = model_1(x1, t1)

        model_0.cleargrads()
        model_1.cleargrads()

        loss_0.backward()
        loss_1.backward()

        model_0.addgrads(model_1)
        optimizer.update()

        model_1.copyparams(model_0)

