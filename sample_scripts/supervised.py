"""
A simple example for supervised learning demonstrating non-differentiable objectives
"""
import torchvision
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from pytorch_optimize.model import Model
from pytorch_optimize.optimizer import ESOptimizer
from pytorch_optimize.objective import Objective, Samples
from sklearn.metrics import f1_score
from torch.optim import Adadelta
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


# objective function
class F1Measure(Objective):
    def __call__(self, model: Model, samples: Samples) -> List[float]:
        outputs = model.forward(samples.inputs)
        predicted_labels = torch.argmax(outputs, dim=1)
        f1_ = f1_score(samples.targets.tolist(), predicted_labels.tolist(), average="macro")
        return [f1_]


if __name__ == "__main__":
    # dataloder
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 50, shuffle=True, num_workers=1)
    trainloader = torch.utils.data.DataLoader(testset, 50, num_workers=1)

    # model
    classifier = Net(n_outputs=10)
    wrapped_model = Model(classifier)

    # objective function (loss function)
    f1_measure = F1Measure()

    # optimizer
    es_optimizer = ESOptimizer(model=wrapped_model, sgd_optimizer=Adadelta(classifier.parameters()),
                               objective_fn=F1Measure(), obj_weights=[1.0], sigma=1e-1, n_samples=5)

    for epoch in range(50):
        epoch_f1_score = 0
        n_batches = 0
        for inputs, targets in tqdm(trainloader):
            samples = BatchSamples(inputs=inputs, targets=targets)
            obj_measure = es_optimizer.gradient_step(samples)
            epoch_f1_score += obj_measure[0]
            n_batches += 1

        print(f"Epoch: {epoch}, Train f1-score: {epoch_f1_score/n_batches}")
