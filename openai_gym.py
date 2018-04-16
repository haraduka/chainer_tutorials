import argparse, os
import gym
from gym import wrappers

import copy, sys
import numpy as np
from collections import deque
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class MLP(Chain):
    def __init__(self, n_in, n_out):
        w = chainer.initializers.HeNormal()
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, 100, initialW=w)
            self.l2 = L.Linear(100, 100,initialW=w)
            self.l3 = L.Linear(100, 100,initialW=w)
            self.l4 = L.Linear(100, n_out, initialW=w)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h


class Agent():
    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_act = n_act
        self.model = MLP(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.mem_size = 1000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20

    def stock_experience(self, st, act, r, st_dash):
        self.memory.append((st, act, r, st_dash))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash):
        Q = self.model(Variable(st))
        tmp = self.target_model(Variable(st_dash))
        max_Q_dash = np.array(map(np.max, tmp.data), dtype=np.float32)
        target = np.asanyarray(Q.data.copy(), dtype=np.float32) # same shape
        for i in xrange(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i])
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        return loss

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash = [], [], [], []
        for i in xrange(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        return st, act, r, st_dash

    def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(xrange(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start+self.batch_size]
            batch = mem[index]
            st, act, r, st_d = self.parse_batch(batch)
            self.model.zerograds()
            loss = self.forward(st, act, r, st_d)
            loss.backward()
            self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act), 0
        else:
            Q = self.model(Variable(st))
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
        self.target_model = copy.deepcopy(self.model)


def main():
    parser = argparse.ArgumentParser(description='OpenAI Gym sample')
    parser.add_argument('--env_name', '-e', type=str, default='CartPole-v0',
            help='Env name: CartPole-v0, Acrobot-v0, MountainCar-v0, Pendulum-v0.')
    parser.add_argument('--monitor', '-m', type=bool, default=True,
            help='Monitor the environment ?')
    parser.add_argument('--resume', '-r', type=bool, default=False,
            help='Resume file from ./model/<env_name>/ ?')
    parser.add_argument('--seed', '-s', type=int, default=0,
            help='random seed.')
    parser.add_argument('--n_episodes', '-n', type=int, default=1000,
            help='Number of episodes.')
    parser.add_argument('--len_episode', '-l', type=int, default=200,
            help='Length of one episode.')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    video_path = "./video/" + args.env_name
    model_path = "./model/" + args.env_name + "_"

    if args.monitor:
        env = wrappers.Monitor(env, "./video/", video_callable=None, force=True)

    # n_st, n_act, actions
    n_st = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v1, MountainCar-v0
        n_act = env.action_space.n
        action_list = range(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = Agent(n_st, n_act, args.seed)
    if args.resume:
        agent.load_model(model_path)

    # main episode loop
    for i_episode in xrange(args.n_episodes):
        observation = env.reset()
        r_sum = 0
        q_sum = 0
        for t in xrange(args.len_episode):
            if args.monitor:
                env.render()
            # decide action
            st = observation.astype(np.float32).reshape((1,n_st))
            act_i, q = agent.get_action(st)
            q_sum += q
            action = action_list[act_i]
            # action!
            observation, reward, done, info = env.step(action)
            if done:
                break
            # evaluate
            st_dash = observation.astype(np.float32).reshape((1,n_st))
            agent.stock_experience(st, act_i, reward, st_dash)
            agent.train()
            r_sum += reward
        print "\t".join(map(str,[i_episode, r_sum, agent.epsilon, agent.loss, q_sum ,agent.step]))
        agent.save_model(model_path)


if __name__=="__main__":
    try:
        os.stat('./model')
    except:
        os.mkdir('./model')
    main()
