#!/bin/sh +x
if [ -e $CONDA_PREFIX_1/envs/bandit_agents_dev_env ]; then
    echo erase bandit_agents_dev_env
    rm -r $CONDA_PREFIX_1/envs/bandit_agents_dev_env
fi

echo creating environment
conda create \
    --name bandit_agents_dev_env \
    --file bandit-agents-dev-env-linux-64.lock -y
