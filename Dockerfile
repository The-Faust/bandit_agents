FROM condaforge/mambaforge:24.3.0-0 as conda

COPY bandit-agents-dev-env-linux-64.lock lib/bandit-agents-dev-env-linux-64.lock

COPY build_env.sh /lib/build_env.sh

RUN chmod +x /lib/build_env.sh

ENV ENV="BUILD"

RUN ./lib/build_env.sh

FROM debian:stable-slim as unit-test-image

COPY --from=conda /env /env

ENV ENV="TEST"

COPY BanditAgents /lib/BanditAgents
COPY Tests /lib/Tests
COPY tox.ini /lib/tox.ini

WORKDIR /lib


CMD ["/env/bin/python3", "-m", "tox", "--current-env"]