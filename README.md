# pytorch-optimize
A simple framework to train pytorch models to optmize for arbitrary non-differentiable objectives.
The gradient is estimated using [Evolution Strategies](https://arxiv.org/pdf/1703.03864.pdf)


## Usage/Steps

1. Wrap your pytorch model with the Model class. The Model class automatically extracts the trainable parameters in the model and samples them at each training step. The sampling strategy can be changed by providing it as a argument to the Model class. 


2. Provide an objective function (a callable) which takes the wrapped model and samples. The objective function then should a scalar value corresponding to the measurement of objective function. The objective function can also return a list of scalar values, in this case, it corresponds to multiple objective functions. 

3. Create an instance for the ESOptimizer


4. Write your training loop


## Examples

Two simple usecases namely reinforcement learning and supervised learning are provided under the following scripts:

1. Reinforcement Learning

2. Supervised learning


## Used in Publications:


## Cite

If you this repository for your research, please 