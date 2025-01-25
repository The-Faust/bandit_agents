#!/bin/sh +x

if [ $ENV = "PUBLISH" ]; then
    /env/bin/python3 -m setup sdist clean --all bdist_wheel &&
    /env/bin/python3 -m twine upload -u __token__ -p $PYPI_TOKEN --verbose dist/*

else
    conda run -n bandit_agents_dev_env python3 setup.py sdist clean --all bdist_wheel
    conda run -n bandit_agents_dev_env python3 -m twine upload -u __token__ -p $PYPI_TOKEN --verbose dist/*
fi