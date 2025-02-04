import os

from setuptools import setup, find_packages


def read(filename) -> str:
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


with open("BanditAgents/requirements.txt") as f:
    requirements: list[str] = f.read().splitlines()

setup(
    name="Bandit Agents",
    author="Vincent Martel",
    author_email="vincent.martel.11235@gmail.com",
    description="Library to solve k-armed bandit problems",
    packages=find_packages(),
    long_description=read("README.md"),
    install_requires=requirements,
    setuptools_git_versioning={"enabled": True},
    setup_requires=[
        "flake8",
        'setuptools',
        'setuptools-git-versioning',
        'wheel',
    ],
)
