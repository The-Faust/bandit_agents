#!/bin/sh +x
conda run -n bandit_agents_dev_env python3 setup.py sdist clean --all bdist_wheel
conda run -n bandit_agents_dev_env python3 -m twine upload -u __token__ -p $PYPI_TOKEN --verbose dist/*