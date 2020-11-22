from setuptools import setup, find_packages

setup(name="pytorch_optimize",
      version="0.0.1",
      description="Package to train pytorch models for non-differentiable objectives",
      author="Rajkumar Ramamurthy",
      author_email="raj1514@gmail.com",
      packages=find_packages(),
      python_requires='>=3.7',
      url="https://github.com/rajcscw/pytorch-optimize",
      install_requires=["numpy", "torch", "torchvision", "tqdm", "pycodestyle", "pytest", "sklearn"])
