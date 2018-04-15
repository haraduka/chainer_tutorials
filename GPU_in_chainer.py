from chainer import cuda
import cupy
import numpy as np

with cupy.cuda.Device(0): # choose the device to use
    x_on_gpu1 = cupy.array([1,2,3,4,5])
print("x_on_gpu1 = " + str(x_on_gpu1))

# cuda.to_gpu and with cupy.cuda.Device is same
x_cpu = np.ones((5,4,3), dtype=np.float32)
x_gpu = cuda.to_gpu(x_cpu, device=0)

x_cpu = np.ones((5,4,3), dtype=np.float32)
with cupy.cuda.Device(0):
    x_gpu = cupy.array(x_cpu)
print("x_gpu = " + str(x_gpu))

# cuda.to_cpu and with x_gpu.device and get()
x_cpu = cuda.to_cpu(x_gpu)

with x_gpu.device:
    x_cpu = x_gpu.get()

cuda.get_device_from_id(0).use()
x_gpu1 = cupy.empty((4, 3), dtype='f') # 'f' indicates float32

with cuda.get_device_from_id(0):
    x_gpu1 = cupy.empty((4, 3), dtype='f')

with cuda.get_device_from_array(x_gpu1):
    y_gpu1 = x_gpu + 1

def add1(x):
    with cuda.get_device_from_array(x):
        return x+1
add1(x_cpu)
add1(x_gpu)

def softplus(x):
    xp = cuda.get_array_module(x)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))
add1(x_cpu)
add1(x_gpu)


# Run Neural Networks on a Single GPU
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

model = L.Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)

model.to_gpu() # here!
batchsize = 100
x_train, y_train = concat_examples(train)
datasize = len(x_train)
for epoch in range(20):
    print('epoch: %d' % epoch)
    indexes = np.random.permutation(datasize)
    for i in range(0, datasize, batchsize):
        x = Variable(cuda.to_gpu(x_train[indexes[i:i+batchsize]])) # here!
        t = Variable(cuda.to_gpu(y_train[indexes[i:i+batchsize]])) # here!
        optimizer.update(model, x, t)




