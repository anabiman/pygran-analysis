# Welcome to the PyGran.analysis webpage!
[//]: # (Badges)
[![CI](https://github.com/Andrew-AbiMansour/PyGranAnalysis/actions/workflows/test.yaml/badge.svg)](https://github.com/Andrew-AbiMansour/PyGranAnalysis/actions/workflows/test.yaml)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/Andrew-AbiMansour/PyGranAnalysis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Andrew-AbiMansour/PyGranAnalysis/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Andrew-AbiMansour/PyGranAnalysis.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Andrew-AbiMansour/PyGranAnalysis/context:python)
[![codecov](https://codecov.io/gh/Andrew-AbiMansour/PyGranAnalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/Andrew-AbiMansour/PyGranAnalysis/branch/master)

PyGranAnalysis is an open-source toolkit primarily designed for analyzing DEM simulation data. It is part of the [PyGran](http://www.pygran.org) ecosystem. 

**If your find PyGran useful in your research, please consider citing the following paper:**

[![DOI for Citing PyGran](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5b00056-blue.svg)](https://doi.org/10.1016/j.softx.2019.01.016)

```
@article{aam2019pygran,
  title={PyGran: An object-oriented library for DEM simulation and analysis},
  author={Abi-Mansour, Andrew},
  journal={SoftwareX},
  volume={9},
  pages={168--174},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.softx.2019.01.016}
}
```

## Quick Installation
PyGranAnalysis is typically installed with other PyGran packages. See [here](http://andrew-abimansour.github.io/PyGran/docs/introduction.html#installation) for more info. For a solo installation, simply run:
```bash
pip install pygran_analysis
```
You can alternatively clone this repository run ``setup.py`` to build and/install the package from its source code. See ``setup.py -h`` for more info.

## Basic Usage
Doing post-analysis with PyGranAnalysis is quite straight forward. Computing particle overlaps shown below for instance can be done in few lines of code:

<p style="text-align:center;"><img src="http://andrew-abimansour.github.io/PyGran/images/overlap-hist.png"></p>

```python
import pygran_analysis

# Instantiate a System class from a dump file
gran = pygran_analysis.System(Particles='granular.dump')

# Instantiate a nearest-neighbors class
nns = analysis.Neighbors(Particles=Gran.Particles)
overlaps = nns.overlaps
```
For more examples on using PyGran for analyzing DEM simulation, check out the <a href="http://andrew-abimansour.github.io/PyGran/tests/examples.html">examples</a> page.
