Installation
############

Installation from pip
=====================

ufs2arco can be installed using pip::

    pip install ufs2arco


Installation from GitHub
========================

To obtain the latest development version, clone
`the repository <https://github.com/NOAA-PSL/ufs2arco>`_
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
