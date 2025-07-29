# ufs2arco
[![Conda version](https://img.shields.io/conda/vn/conda-forge/ufs2arco.svg)](https://anaconda.org/conda-forge/ufs2arco)
[![PyPI version](https://img.shields.io/pypi/v/ufs2arco.svg)](https://pypi.org/project/ufs2arco/)
![OS Support](https://img.shields.io/badge/OS-Linux%20%7C%20macOS-blue?)
![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue?logo=python&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/ufs2arco/badge/?version=latest)](https://ufs2arco.readthedocs.io/en/latest/?badge=latest)

`ufs2arco` is a python package that is designed to make [NOAA](https://www.noaa.gov/)
forecast, reanalysis, and reforecast datasets
more accessible for scientific analysis and machine learning model development.
The name stems from its original intent, which was to transform output from the
[Unified Forecast System (UFS)](https://www.ufs.epic.noaa.gov/)
into
Analysis Ready, Cloud Optimized (ARCO;
[Abernathey et al., (2021)](https://doi.ieeecomputersociety.org/10.1109/MCSE.2021.3059437))
format.
However, the package now pulls data from a number of non-UFS sources, including GFS/GEFS
before UFS was created, and even
[ECMWF's ERA5 dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).

To learn how to use `ufs2arco`, check out the
[documentation here](https://ufs2arco.readthedocs.io/en/latest/index.html)


## Installation

### Recommended: Install from conda-forge

Given that some `ufs2arco` dependencies are only available on conda-forge, it's
recommended that users install using this method

```shell
conda install -c conda-forge ufs2arco
```

This will install all underlying dependencies, including
[mpi4py](https://mpi4py.readthedocs.io/en/latest/index.html)
which by default installs [mpich](https://pypi.org/project/mpich) for MPI
support.
If a different MPI distribution is desired, for example
[Intel MPI](https://pypi.org/project/impi-rt/), then this can be installed via:

```shell
conda install -c conda-forge ufs2arco impi_rt
```

in the exact same way as directed in
[this section of the mpi4py documentation](https://mpi4py.readthedocs.io/en/latest/install.html#conda-packages).

### Install from conda-forge without MPI

If you do not want to install MPI through conda-forge, for instance if you want
to use an MPI distribution that is already built on a system you're using, then it is
recommended to install the `nompi` build from conda-forge as follows:

```shell
conda install -c conda-forge ufs2arco=*=nompi*
```

Then, one can install mpi4py from pip using [these
instructions](https://mpi4py.readthedocs.io/en/latest/install.html#building-from-sources),
or following instructions specific to your machine.

### Install from pip

It is possible to install ufs2arco from pypi via:

```bash
pip install ufs2arco
```

However, this will not come with the MPI or
[xesmf](https://xesmf.readthedocs.io/en/stable/) dependencies, since these need
to be installed from conda-forge.


## Get in touch

Report bugs, suggest features, or view the source code
[on GitHub](https://github.com/NOAA-PSL/ufs2arco).

## License and Copyright

`ufs2arco` is licensed under the Apache-2.0 License.

Development occurs on GitHub at <https://github.com/NOAA-PSL/ufs2arco>.
