FROM continuumio/miniconda3
MAINTAINER Michael Ekstrand <md@ekstrandom.net>

RUN conda config --set always_yes yes --set changeps1 no
RUN conda install -q python=3.6 && conda update -q --all && conda clean -t
RUN conda install -q pandas scipy cython pytables && conda clean -t
RUN conda install -q pytest pytest-arraydiff pytest-xdist invoke && conda clean -t

RUN groupadd --gid 3434 circleci \
  && useradd --uid 3434 --gid circleci --shell /bin/bash --create-home circleci
USER circleci
CMD ["/bin/bash"]
