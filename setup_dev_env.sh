#!/bin/sh +x

chmod +x build_env.sh compile_env_lock.sh

if [ ! -e bandit-agents-dev-env-linux-64.lock ] || [ $1 ];
then
  ./compile_env_lock.sh 1
fi

./build_env.sh

conda run -n bandit_agents_dev_env pre-commit install