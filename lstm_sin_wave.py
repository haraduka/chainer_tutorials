import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import matplotlib.pyplot as plt

class MLP(Chain):
    def __init__(self, n_input, n_units, n_output):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_output)

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return self.l3(h2)

class LossFunc(Chain):
    def __init__(self, predictor):
        super(LossFunc, self).__init__()
        with self.init_scope():
            self.predictor=predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        #loss = F.mean_absolute_error(y, t)
        loss = F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss

class LSTMiterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size=10, seq_len=5, repeat=True):
        self.seq_len = seq_len
        self.dataset = dataset
        self.nsamples = len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat
        self.epoch = 0
        self.iteration = 0
        self.offsets = np.random.randint(0, self.nsamples, size=batch_size)
        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
            raise StopIteration
        x, t = self.get_data()
        x = np.array(x).reshape((-1, 1))
        t = np.array(t).reshape((-1, 1))
        self.iteration += 1
        epoch = self.iteration // self.batch_size
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            self.offsets = np.random.randint(0, self.nsamples, size=self.batch_size)
        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / self.nsamples

    def get_data(self):
        tmp0 = [self.dataset[(offset + self.iteration)%self.nsamples][0]
                for offset in self.offsets]
        tmp1 = [self.dataset[(offset + self.iteration + 1)%self.nsamples][0]
                for offset in self.offsets]
        return tmp0, tmp1

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

class LSTMupdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(LSTMupdater, self).__init__(train_iter, optimizer, device=device)
        self.seq_len = train_iter.seq_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        for i in range(self.seq_len):
            batch = np.array(train_iter.__next__()).astype(np.float32)
            x, t = batch[:, 0], batch[:, 1]
            loss += optimizer.target(x, t)
        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

def main():
    model = LossFunc(MLP(1, 5, 1))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    N_data = 100
    N_loop = 3
    t = np.linspace(0, 2*np.pi*N_loop, num=N_data)
    X = 0.8*np.sin(2.0*t)
    N_train = int(N_data*0.8)
    N_test = N_data-N_train
    tmp_dataset_x = np.array(X).astype(np.float32)
    x_train, x_test = np.array(tmp_dataset_x[:N_train]), np.array(tmp_dataset_x[N_train:])

    train = tuple_dataset.TupleDataset(x_train)
    test = tuple_dataset.TupleDataset(x_test)

    train_iter = LSTMiterator(train, batch_size=10, seq_len=10)
    test_iter = LSTMiterator(test, batch_size=10, seq_len=10, repeat=False)

    updater = LSTMupdater(train_iter, optimizer, -1)
    trainer = training.Trainer(updater, (1000, 'epoch'), out='result')

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=-1, eval_hook=lambda _: eval_rnn.reset_state()))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    # prediction 1
    #presteps = 10
    #model.predictor.reset_state()
    #for i in range(presteps):
    #    y = model.predictor(np.roll(x_train, i).reshape((-1, 1)))
    #plt.plot(t[:N_train], np.roll(y.data, -presteps))
    #plt.plot(t[:N_train], x_train)
    #plt.show()

    # prediction 2
    presteps = int(N_data*0.1)
    poststeps = N_data-presteps
    model.predictor.reset_state()
    y_result = []
    for i in range(presteps):
        y = model.predictor(x_train[i].reshape((-1, 1)))
        y_result.append(x_train[i])

    for i in range(poststeps):
        y = model.predictor(y.data)
        y_result.append(y.data)
    plt.plot(t, y_result)
    plt.plot(t, X)
    plt.show()

if __name__ == '__main__':
    main()
