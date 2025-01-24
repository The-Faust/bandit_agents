import os

from setuptools import setup, find_packages


def read(filename) -> str:
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


with open("BanditAgents/requirements.txt") as f:
    requirements: list[str] = f.read().splitlines()

setup(
    name="Bandit Problem Solvers",
    version="0.5.0",
    author="Vincent Martel",
    author_email="vincent.martel.11235@gmail.com",
    description="Library to solve k-armed bandit problems",
    packages=find_packages(exclude=["Tests"]),
    long_description=read("BanditAgents/README.md"),
    install_requires=requirements,
    setup_requires=["flake8", 'setuptools', 'setuptools-git', 'wheel'],
)
