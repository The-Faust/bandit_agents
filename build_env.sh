#!/bin/sh +x

if [ $ENV = "BUILD" ]; then
  mamba create \
    --copy -p ./env \
    --file /lib/bandit-agents-dev-env-linux-64.lock \
    && conda clean -afy

  find -name '*.a' -delete &&
    rm -rf /env/conda-meta &&
    rm -rf /env/include &&
    rm /env/lib/libpython3.12.so.1.0 &&
    find -name '__pycache__' -type d -exec rm -rf '{}' '+' &&
    rm -rf /env/lib/python3.12/site-packages/pip /env/lib/python3.12/idlelib /env/lib/python3.12/ensurepip \
      /env/lib/libasan.so.5.0.0 \
      /env/lib/libtsan.so.0.0.0 \
      /env/lib/liblsan.so.0.0.0 \
      /env/lib/libubsan.so.1.0.0 \
      /env/bin/x86_64-conda-linux-gnu-ld \
      /env/bin/sqlite3 \
      /env/bin/openssl \
      /env/share/terminfo &&
    find /env/lib/python3.12/site-packages/scipy -name 'tests' -type d -exec rm -rf '{}' '+' &&
    find /env/lib/python3.12/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' &&
    find /env/lib/python3.12/site-packages/pandas -name 'tests' -type d -exec rm -rf '{}' '+' &&
    find /env/lib/python3.12/site-packages -name '*.pyx' -delete &&
    rm -rf /env/lib/python3.12/site-packages/uvloop/loop.c

else
  if [ -e $CONDA_PREFIX_1/envs/bandit_agents_dev_env ]; then
    echo erase bandit_agents_dev_env
    rm -r $CONDA_PREFIX_1/envs/bandit_agents_dev_env
  fi

  echo creating environment
  conda create \
    --name bandit_agents_dev_env \
    --file bandit-agents-dev-env-linux-64.lock -y

  echo adding dev dependencies
  conda update \
    --name bandit_agents_dev_env \
    --freeze-installed \
    --no-update-deps \
    --file bandit-agents-dev-env-linux-64.lock
fi
