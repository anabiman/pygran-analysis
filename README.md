# Welcome to the PyGran.analysis webpage!

PyGran.analysis is a submodule in PyGran, an open-source toolkit primarily designed for analyzing DEM simulation data. You can read more about PyGran [here](http://www.pygran.org). 

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
PyGran.analysis is typically installed with other PyGran submodules. See [here](http://andrew-abimansour.github.io/PyGran/docs/introduction.html#installation) for more info. For a solo PyGran.analysis local installation, simply clone this repository and then use pip (or pip3) to run from the source dir:
```bash
pip install . --user
```
You can alternatively run ``setup.py`` to build and/install the package. See ``setup.py -h`` for more info.

## Basic Usage
Using PyGran.analysis for doing post-analysis is quite straight forward. Computing particle overlaps shown below for instance can be done in few lines of code:

<p style="text-align:center;"><img src="http://andrew-abimansour.github.io/PyGran/images/overlap-hist.png"></p>

```python
import PyGran.analysis

# Instantiate a System class from a dump file
Gran = PyGran.analysis.System(Particles='granular.dump')

# Instantiate a nearest-neighbors class
NNS = PyGran.analysis.Neighbors(Particles=Gran.Particles)
overlaps = NNS.overlaps
```
For more examples on using PyGran for analyzing DEM simulation, check out the <a href="http://andrew-abimansour.github.io/PyGran/tests/examples.html">examples</a> page.
