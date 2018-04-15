# 1. Prepare a dataset
from __future__ import print_function
import matplotlib.pyplot as plt
from chainer.datasets import mnist

train, test = mnist.get_mnist(withlabel=True, ndim=1)

x,t = train[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.savefig('5.png')
print('label: ', t)


# 2. Create a dataset iterator
from chainer import iterators
batchsize = 128
# train = [(x1, t1), (x2, t2)] # A list of tuples
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


# 3. Define a network
from chainer import Chain
import chainer.links as L
import chainer.functions as F
class MyNetwork(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

model = MyNetwork()


# 4. Select an optimization algorithm
from chainer import optimizers
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)


# 5. Write a training loop
import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu, to_gpu

max_epoch = 10
gpu_id = 0

model.to_gpu()

while train_iter.epoch < max_epoch:
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)
    prediction_train = model(image_train)
    loss = F.softmax_cross_entropy(prediction_train, target_train)
    model.cleargrads()
    loss.backward()
    optimizer.update()

    if train_iter.is_new_epoch:
        print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(to_cpu(loss.data))), end='')
        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch, gpu_id)
            prediction_test = model(image_test)
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(to_cpu(loss_test.data))
            accuracy = F.accuracy(prediction_test, target_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break
        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))


# 6. Save the trained model
from chainer import serializers
serializers.save_npz('my_mnist.model', model)


# 7. Perform classification by the saved model
model = MyNetwork()
serializers.load_npz('my_mnist.model', model)
x, t = test[0]
plt.imshow(x.reshape((28, 28)), cmap='gray')
plt.savefig('7.png')
print('label:', t)
