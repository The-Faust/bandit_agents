#!/bin/sh +x

hash conda 2>/dev/null || { echo >&2 "I require conda but it's not installed.  Aborting."; exit 1; }
conda lock --version 2>/dev/null || { echo >&2 "I require conda-lock but it's not installed. Aborting."; exit 1; }

if [ ! -e bandit-agents-dev-env-linux-64.lock ] || 
  [ ! -e bandit-agents-dev-env-win-64.lock ] || 
  [ ! -e bandit-agents-dev-env-osx-64.lock ] || 
  [ $1 ];
then
  echo Compiling the conda env lock file this could take a while \\n

  conda lock \
    -f bandit_agents_dev_env.yml \
    -k explicit \
    --filename-template "bandit-agents-dev-env-{platform}.lock"
fi