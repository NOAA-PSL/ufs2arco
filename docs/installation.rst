Installation
############

Recommended: Install from conda-forge
=====================================

Given that some ``ufs2arco`` dependencies are only available on conda-forge, it's
recommended that users install using this method::

    conda install -c conda-forge ufs2arco

This will install all underlying dependencies, including
`mpi4py <https://mpi4py.readthedocs.io/en/latest/index.html>`_
which by default installs `mpich <https://pypi.org/project/mpich>`_
for MPI support.
If a different MPI distribution is desired, for example
`Intel MPI <https://pypi.org/project/impi-rt/>`_,
then this can be installed via::

    conda install -c conda-forge ufs2arco impi_rt

in the exact same way as directed in
`this section of the mpi4py documentation <https://mpi4py.readthedocs.io/en/latest/install.html#conda-packages>`_.

Install from conda-forge without MPI
====================================

In order to install and use MPI that is prebuilt on your system, for example on
an HPC machine, then it is recommended to install the ``ufs2arco-nompi`` build on
conda forge as follows::

    conda install -c conda-forge ufs2arco-nompi

Then, install mpi4py from pip using
`these instructions <https://mpi4py.readthedocs.io/en/latest/install.html#building-from-sources>`_,
or following instructions specific to your machine.


Install with pip
================

It is possible to install ufs2arco from pypi via::

    pip install ufs2arco

However, this will not come with the MPI or xesmf dependencies, since these need
to be installed via conda-forge.

Install from GitHub
===================

To obtain the latest development version, clone
`the repository <https://github.com/NOAA-PSL/ufs2arco>`_.
and install it as follows::

    git clone https://github.com/NOAA-PSL/ufs2arco.git
    cd ufs2arco
    pip install -e .

Users are encourged to `fork <https://help.github.com/articles/fork-a-repo/>`_
the project and submit 
`issues <https://github.com/NOAA-PSL/ufs2arco/issues>`_
and
`pull requests <https://github.com/NOAA-PSL/ufs2arco/pulls>`_.

Running Example Notebooks or Building the Documentation Locally
===============================================================

Due to the way pandoc is installed via pip `as detailed here
<https://stackoverflow.com/a/71585691>`_
it is recommended to create an environment with conda in order to build the
documentation locally.
This is also recommended for running any of the example notebooks locally, since
there are a couple of additional dependencies required.
To do this, first download `this environment.yaml file
<https://github.com/NOAA-PSL/ufs2arco/blob/main/environment.yaml>`_,
then create the conda environment::

    conda env create -f environment.yaml
    conda activate ufs2arco

Note that you will then want to add ufs2arco to the jupyter kernel::

    python -m ipykernel install --user --name=ufs2arco

so that it can be accessed from jupyter lab / notebook.

