# pytorch-optimize [![CircleCI](https://circleci.com/gh/rajcscw/pytorch-optimize/tree/master.svg?style=svg)](https://circleci.com/gh/rajcscw/pytorch-optimize/tree/master)

pytorch-optimize is a simple black-box framework to train pytorch models for optimizing arbitrary objective functions. It provides simple wrappers for models and optimizers so that they can be used to optimize the provided objective function (including non-differentiable objectives). It also supports optimization of multiple objectives out-of-the-box. The optimizer itself is based on
[Evolution strategies](https://arxiv.org/pdf/1703.03864.pdf) which estimates gradient using parallel workers so that it can scale well utilizing multiple cores.

## Install
```
git clone https://github.com/rajcscw/pytorch-optimize.git
cd pytorch-optimize
pip install .
```

## Usage

 1.Wrap your pytorch model (`torch.nn.Module`) using the [`Model`](https://github.com/rajcscw/pytorch-optimize/blob/master/pytorch_optimize/model.py)  class. The Model class automatically extracts the trainable parameters in the network and samples them at each training step. The sampling strategy can be changed by providing it as an argument to the Model class. Possible strategies include sampling layers from bottom to up, top to bottom, random or all the layers at once.

```python
from pytorch_optimize.model import Model, SamplingStrategy
net = Net(..)
model = Model(net=net, strategy=SamplingStrategy.BOTTOM_UP)
```

2.Provide an [`objective function`](https://github.com/rajcscw/pytorch-optimize/blob/master/pytorch_optimize/objective.py) (a callable) which takes the wrapped model and [`samples`](https://github.com/rajcscw/pytorch-optimize/blob/master/pytorch_optimize/objective.py) as its inputs. The objective function then should return a scalar value corresponding to the measurement of objective function. The objective function can also return a list of scalar values, in this case, it corresponds to multiple objective functions.
Note, here [`Samples`](https://github.com/rajcscw/pytorch-optimize/blob/master/pytorch_optimize/objective.py) is just a simple dataclass for wrapping data for computing the objective function. For instance, in supervised learning, it contains inputs and targets. For reinforcement learning, this could be environments, seeds etc. 


```python
from pytorch_optimize.objective import Objective, Samples

class MyObjective(Objective):
    def __call__(self, model: Model, samples: Samples) -> List[float]
        # compute your objective function(s)
        return objectives

my_objective = MyObjective() 
```

3.Create an instance of the [`ESOptimizer`](https://github.com/rajcscw/pytorch-optimize/blob/master/pytorch_optimize/optimizer.py). This takes an instance of the wrapped model, SGD optimizer and the objective function. Additionally, you have to pass weights corresponding to each of the objective functions using `obj_weights`. Further, parameters `sigma` and `n_samples` for ES have to be passed. Internally, the objectives are subject to rank transformation so the scales of objective function(s) does not influence the optimization.

Note: The optimizer does gradient ascent instead of descent. Therefore, the objective functions needs to be implemented accordingly(for instance, returning 1/loss instead of loss).

```python
sgd_optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
es_optimizer = ESOptimizer(model=model, sgd_optimizer=sgd_optimizer, objective_fn=my_objective, 
                           obj_weights=[1.0],sigma=1e-1, n_samples=100)
```

4.Write your usual training loop or trainer routine with the following template. 

```python
for epoch in range(1000):
    samples = Samples(..)                               # wrap data
    es_optimizer.gradient_step(samples)                 # gradient step
    objective_at_epoch = MyObjective()(model, samples)  # measure objective after stepping
```

## Demo scripts

Two simple showcases: reinforcement learning and supervised learning are provided in the [sample_scripts](https://github.com/rajcscw/pytorch-optimize/tree/master/sample_scripts) folder:

1.**Supervised Learning:** As an illustrative example, [supervised.py](https://github.com/rajcscw/pytorch-optimize/blob/master/sample_scripts/supervised.py) shows training a classifier to classify MNIST digits by directly optimizing the accuracy rather than cross-entropy loss.


2.**Reinforcement Learning:** 
Similary, the script [rl.py](https://github.com/rajcscw/pytorch-optimize/blob/master/sample_scripts/rl.py) shows how to train an RL agent that tries to maximize the episodic reward it receives while solving the task cart pole balancing task. To run this script, install also [gym](https://github.com/openai/gym).

## Contributions
You are welcome to contribute to the repository by developing new features or fixing bugs. If you do so, please create a pull request.

## Cite

If you use this repository for your research, please cite with the following bibtex:

```
@misc{ramamurthy2020pytorch_optimize,
  title = {pytorch-optimize - a black box optimization framework},
  author = {Ramamurthy, Rajkumar},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rajcscw/pytorch-optimize}},
}
```
