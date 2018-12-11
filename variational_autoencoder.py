import argparse
import os

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.distributions as D
from chainer import training, datasets, iterators, optimizers, reporter
from chainer.training import extensions


class AvgELBOLoss(chainer.Chain):
    def __init__(self, encoder, decoder, prior, beta=1.0, k=1):
        super(AvgELBOLoss, self).__init__()
        self.beta = beta
        self.k = k
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.prior = prior

    def __call__(self, x):
        q_z = self.encoder(x)
        z = q_z.sample(self.k)
        p_x = self.decoder(z)
        p_z = self.prior()

        reconstr = F.mean(F.sum(p_x.log_prob(F.broadcast_to(x[None, :], (self.k,) + x.shape)), axis=-1))
        kl_penalty = F.mean(F.sum(chainer.kl_divergence(q_z, p_z), axis=-1))
        loss = - (reconstr - self.beta * kl_penalty)
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        return loss


class Encoder(chainer.Chain):
    def __init__(self, n_in, n_h, n_latent):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(n_in, n_h)
            self.fc2_ave = L.Linear(n_h, n_latent)
            self.fc2_var = L.Linear(n_h, n_latent)

    def __call__(self, x):
        h = F.tanh(self.fc1(x))
        ave = self.fc2_ave(h)
        var = self.fc2_var(h)  # log(sigma)
        return D.Normal(loc=ave, log_scale=var)


class Decoder(chainer.Chain):
    def __init__(self, n_in, n_h, n_latent):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(n_latent, n_h)
            self.fc2 = L.Linear(n_h, n_in)

    def __call__(self, z, inference=False):
        n_batch_axes = 1 if inference else 2
        h = F.tanh(self.fc1(z, n_batch_axes=n_batch_axes))
        h = self.fc2(h, n_batch_axes=n_batch_axes)
        return D.Bernoulli(logit=h)


class Prior(chainer.Link):
    def __init__(self, n_latent):
        super(Prior, self).__init__()

        self.loc = np.zeros(n_latent, np.float32)
        self.scale = np.ones(n_latent, np.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def __call__(self):
        return D.Normal(self.loc, scale=self.scale)


def main():
    parser = argparse.ArgumentParser(description='variational auto encoder')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help="gpu ID")
    parser.add_argument('--epoch', '-e', default=100, type=int, help="epoch num")
    parser.add_argument('--batch', '-b', default=100, type=int, help="batch size")
    parser.add_argument('--dimz', '-z', default=20, type=int, help="latent space size")
    parser.add_argument('--dimh', default=500, type=int, help="hidden layer size")
    parser.add_argument('--k', '-k', default=1, type=int, help="for monte carlo sampling")
    parser.add_argument('--beta', default=1.0, type=float, help="weight")
    parser.add_argument('--out', '-o', default="result", type=str, help="output directory")
    args = parser.parse_args()

    encoder = Encoder(784, args.dimh, args.dimz)
    decoder = Decoder(784, args.dimh, args.dimz)
    prior = Prior(args.dimz)

    avg_elbo_loss = AvgELBOLoss(encoder, decoder, prior, beta=args.beta, k=args.k)

    if args.gpu >= 0:
        avg_elbo_loss.to_gpu(args.gpu)

    optimizer = optimizers.Adam()
    optimizer.setup(avg_elbo_loss)

    train, test = datasets.get_mnist(withlabel=False)

    train_iter = iterators.SerialIterator(train, args.batch)
    test_iter = iterators.SerialIterator(test, args.batch, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu, loss_func=avg_elbo_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, avg_elbo_loss, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'varidation/main/loss', 'main/reconstr', 'main/kl_penalty', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    def save_images(x, filename):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(28, 28))
        fig.savefig(filename)

    avg_elbo_loss.to_cpu()

    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = decoder(encoder(x).mean, inference=True).mean
    save_images(x.array, os.path.join(args.out, 'train'))
    save_images(x1.array, os.path.join(args.out, 'train_reconstructed'))

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = decoder(encoder(x).mean, inference=True).mean
    save_images(x.array, os.path.join(args.out, 'test'))
    save_images(x1.array, os.path.join(args.out, 'test_reconstructed'))

    z = prior().sample(9)
    x = decoder(z, inference=True).mean
    save_images(x.array, os.path.join(args.out, 'sampled'))


if __name__ == '__main__':
    main()
