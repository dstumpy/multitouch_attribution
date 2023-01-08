# syntax=docker/dockerfile:experimental
# BASE
FROM python:3.10-slim-bullseye AS base

# install UNIX tools
RUN apt-get update \
    && apt-get install -y apt-transport-https curl git gnupg2 openssh-server \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y unixodbc-dev

# Set timezone in container
ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

# install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# source poetry
# https://stackoverflow.com/questions/59895745/poetry-fails-to-install-in-docker
ENV PATH = "${PATH}:/root/.poetry/bin:/root/.local/bin"

# create app folder & add to pythonpath for direct python shell execution
WORKDIR /home/root/app
ENV PYTHONPATH=/home/root/app

FROM base as manual

# add auto-complete for git
RUN /bin/bash -c "curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash"; \
    /bin/bash -c "echo 'source ~/.git-completion.bash' >> ~/.bashrc"

# DEV with pyproject.toml and optional lock-file
FROM manual as dev
# Create VsCode extension folder for consistent volumne mount
# https://code.visualstudio.com/docs/remote/containers-advanced#_avoiding-extension-reinstalls-on-container-rebuild
RUN mkdir -p /root/.vscode-server/extensions

# IDE
FROM manual as ide

# add poetry PATH to PATH variable, since
# code-server:4.1.0 overwrites PATH
# issue: https://gihub.com/coder/code-server/issues/4699
RUN echo 'export PATH=$PATH:/root/.local/bin' >> ~/.bashrc

# install code-server
RUN curl -fOL https://github.com/coder/code-server/releases/download/v4.1.0/code-server_4.1.0_amd64.deb; \
    dpkg -i code-server_4.1.0_amd64.deb
# extensions to code-server
RUN code-server --install-extension ms-python.python \
    code-server --install-extension ms-pyright.pyright \
    code-server --install-extension mhutchie.git-graph \
    code-server --install-extension njpwerner.autodocstring \
    code-server --install-extension streetsidesoftware.code-spell-checker

# download resource monitor manually, since mutantdino.resourcemonitor cannot be loaded with code-server
RUN curl  -o resmon.vsix.gz https://marketplace.visualstudio.com/_apis/public/gallery/publishers/mutantdino/vsextensions/resourcemonitor/1.0.7/vspackage; \
    gunzip resmon.vsix.gz; \
    code-server --install-extension ./resmon.vsix \
    rm resmon.vsix;

# install poetry packages from .lock file
COPY pyproject.toml poetry.lock* ./
RUN poetry install &&\
    rm pyproject.toml poetry.lock

# PROD with pyproject.toml and lock-file
FROM base as prod
COPY . .
RUN poetry install --no-dev
