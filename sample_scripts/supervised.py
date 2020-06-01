"""
A simple example for supervised learning demonstrating non-differentiable objectives
"""
import torchvision
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from pytorch_optimize.model import Model, SamplingStrategy
from pytorch_optimize.optimizer import ESOptimizer
from pytorch_optimize.objective import Objective, Samples
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam, Adadelta
from dataclasses import dataclass
from typing import List
from tqdm import tqdm


@dataclass(init=True)
class BatchSamples(Samples):
    inputs: torch.Tensor
    targets: torch.Tensor


# set up the network architecture
class Net(nn.Module):
    def __init__(self, n_outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Accuracy(Objective):
    def __call__(self, model: Model, samples: Samples, device: str) -> List[float]:
        deivce = torch.device(device)
        outputs = model.forward(samples.inputs.to(device))
        predicted_labels = torch.argmax(outputs, dim=1).cpu()
        accuracy = accuracy_score(samples.targets.tolist(), predicted_labels.tolist())
        return [accuracy]


class CELoss(Objective):
    def __call__(self, model: Model, samples: Samples, device: str) -> List[float]:
        deivce = torch.device(device)
        outputs = model.forward(samples.inputs.to(device))
        loss = torch.nn.CrossEntropyLoss()(outputs, samples.targets.to(device))
        return [1/loss]


if __name__ == "__main__":
    # dataloder
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 50, shuffle=True, num_workers=1)
    trainloader = torch.utils.data.DataLoader(testset, 50, num_workers=1)

    # model
    classifier = Net(n_outputs=10)
    wrapped_model = Model(classifier, strategy=SamplingStrategy.ALL)

    # objective function (loss function)
    obj_measure = CELoss()

    # optimizer
    es_optimizer = ESOptimizer(model=wrapped_model, sgd_optimizer=Adam(classifier.parameters()),
                               objective_fn=obj_measure, obj_weights=[1.0], sigma=1e-1, n_samples=50,
                               devices=["cpu"], n_workers=10)

    for epoch in range(5):
        n_batches = 0
        running_accuracy = 0
        show_every = 50
        for i, (inputs, targets) in enumerate(tqdm(trainloader)):
            samples = BatchSamples(inputs=inputs, targets=targets)
            es_optimizer.gradient_step(samples)

            accuracy = Accuracy()(wrapped_model, samples, "cpu")
            n_batches += 1
            running_accuracy += accuracy[0]

            if (i+1) % show_every == 0:
                print(f"Iter {i}, Running accuracy: {running_accuracy/n_batches}")
                n_batches = 0
                running_accuracy = 0
