import os

from setuptools import setup, find_packages


def read(filename) -> str:
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="Bandit Agents",
    description="Library to solve k-armed bandit problems",
    packages=find_packages(),
    long_description=read("./README.md"),
    setuptools_git_versioning={"enabled": True},
    setup_requires=[
        "flake8",
        'setuptools',
        'setuptools-git-versioning',
        'wheel',
    ],
)
