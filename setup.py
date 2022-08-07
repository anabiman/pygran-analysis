"""
Setup file for pygran_analysis

Created on October 10, 2019

Author: Andrew Abi-Mansour

This is the::

        ██████╗ ██╗   ██╗ ██████╗ ██████╗  █████╗ ███╗   ██╗
        ██╔══██╗╚██╗ ██╔╝██╔════╝ ██╔══██╗██╔══██╗████╗  ██║
        ██████╔╝ ╚████╔╝ ██║  ███╗██████╔╝███████║██╔██╗ ██║
        ██╔═══╝   ╚██╔╝  ██║   ██║██╔══██╗██╔══██║██║╚██╗██║
        ██║        ██║   ╚██████╔╝██║  ██║██║  ██║██║ ╚████║
        ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝

DEM simulation and analysis toolkit
http://www.pygran.org, support@pygran.org

Core developer and main author:
Andrew Abi-Mansour, andrew.abi.mansour@pygran.org

PyGran is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENS
"""

import os
import warnings

from setuptools import find_packages, setup

import versioneer

try:
    import numpy
    from Cython.Build import cythonize

    optimal_list = cythonize(
        "pygran_analysis/opt_core.pyx",
        compiler_directives={"language_level": "3"},
        annotate=True,
    )
    include_dirs = [numpy.get_include()]
except Exception:
    warnings.warn(
        "Could not cythonize. Make sure Cython and NumPy are properly installed. "
        "Proceeding with unoptimized code ...",
        UserWarning,
    )
    optimal_list = []
    include_dirs = []

setup(
    name="pygran_analysis",
    version=versioneer.get_version(),
    author="Andrew Abi-Mansour",
    author_email="support@pygran.org",
    description=(
        "A package for rapid quantitative analysis of granular/powder systems"
    ),
    license="GNU v2",
    keywords="Discrete Element Method, Granular Materials",
    url="http://pygran.org",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={"extras": ["vtk", "cython"], "tests": ["pytest"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Cython",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
    ext_modules=optimal_list,
    include_dirs=include_dirs,
    cmdclass=versioneer.get_cmdclass(),
)
