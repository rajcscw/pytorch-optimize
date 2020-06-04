"""
A sample script to train an RL agent
"""
from torch import nn
import torch.nn.functional as F
from pytorch_optimize.model import Model, SamplingStrategy
from pytorch_optimize.optimizer import ESOptimizer
from pytorch_optimize.objective import Objective, Samples
from torch.optim import SGD, Adam, Adadelta
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import gym
import torch


@dataclass(init=True)
class EnvSamples(Samples):
    env_name: str


# set up the network for a softmax policy
class Net(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        return x


class EpisodicReturn(Objective):
    def __call__(self, model: Model, samples: Samples, device: str = "cpu") -> List[float]:
        # play an episode
        env = gym.make(samples.env_name)
        env._max_episode_steps = 500
        state = env.reset()
        done = False
        episodic_return = 0
        while not done:
            state = torch.from_numpy(state).reshape((1, -1)).float()
            action = int(model.forward(state))
            state, reward, done, info = env.step(action)
            episodic_return += reward
        return [episodic_return]


if __name__ == "__main__":
    # model
    policy = Net(n_inputs=4, n_hidden=50, n_outputs=2)
    wrapped_model = Model(policy, strategy=SamplingStrategy.BOTTOM_UP)

    # objective function (loss function)
    reward_measure = EpisodicReturn()

    # optimizer
    es_optimizer = ESOptimizer(model=wrapped_model, sgd_optimizer=SGD(policy.parameters(), lr=1e-2),
                               objective_fn=reward_measure, obj_weights=[1.0],sigma=1e-1, n_samples=10,
                               devices=["cpu"])

    # create env samples
    samples = EnvSamples(env_name="CartPole-v0")

    # train for number of epochs
    running_return = 0
    show_every = 50
    for epoch in tqdm(range(1000)):
        es_optimizer.gradient_step(samples)
        running_return += EpisodicReturn()(wrapped_model, samples)[0]
        if (epoch+1) % show_every == 0:
            print(f"Iter {epoch}, Running return: {running_return/show_every}")
            running_return = 0
