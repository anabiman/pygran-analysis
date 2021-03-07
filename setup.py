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

from setuptools import setup, find_packages
import versioneer
import warnings
import os

try:
    from Cython.Build import cythonize
    import numpy

    optimal_list = cythonize("pygran_analysis/core.pyx")
    include_dirs = [numpy.get_include()]
except Exception:
    warnings.warn(
        "Could not cythonize. Make sure Cython is properly installed. Proceeding with unoptimized code.",
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
    url="www.PyGran.org",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={"extras": ["vtk", "Pillow"], "tests": ["pytest"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Cython",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
    ext_modules=optimal_list,
    include_dirs=include_dirs,
    cmdclass=versioneer.get_cmdclass(),
)
