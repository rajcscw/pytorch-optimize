from setuptools import setup, find_packages

setup(name="pytorch_optimize",
      version="0.0.1",
      description="Package providing support to train pytorch models",
      author="Rajkumar Ramamurthy",
      author_email="raj1514@gmail.com",
      packages=find_packages(),
      install_requires=["numpy", "torch", "torchvision", "tqdm", "pycodestyle", "pytest", "sklearn", "gym"])
