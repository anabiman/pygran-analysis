'''
	Created on July 9, 2016
	@author: Andrew Abi-Mansour

	This is the 
	 __________         ________                     
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
	and LICENSE files.
 '''

from setuptools import setup, find_packages
import glob, shutil, re
from distutils.command.install import install
from distutils.command.clean import clean
from _version import __version__, __author__, __email__

try:
	from Cython.Build import cythonize
	import numpy
	optimal_list = cythonize("src/core.pyx")
	include_dirs = [numpy.get_include()]
except:
	print('Could not cythonize. Make sure Cython is properly installed.')
	optimal_list = []
	include_dirs = []

setup(
		name = "PyGran.analysis",
		version = __version__,
		author = __author__,
		author_email = __email__,
		description = ("A DEM toolkit for rapid quantitative analysis of granular/powder systems"),
		license = "GNU v2",
		keywords = "Discrete Element Method, Granular Materials",
		url = "https://github.com/Andrew-AbiMansour/PyGran",
		packages=find_packages('.'),
		include_package_data=True,
		install_requires=['numpy', 'scipy'],
		extras_require={'extra': ['vtk', 'Pillow']},
		long_description='A DEM toolbox for rapid quantitative analysis of granular/powder systems. See http://www.pygran.org.',
		classifiers=[
				"Development Status :: 4 - Beta",
				"Topic :: Utilities",
				"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
				"Programming Language :: Python :: 2.7",
				"Programming Language :: Python :: 3",
				"Programming Language :: Python :: 3.4",
				"Programming Language :: Python :: 3.5",
				"Programming Language :: Python :: 3.6",
				"Programming Language :: Python :: 3.7",
				"Programming Language :: C",
				"Operating System :: POSIX :: Linux"
		],

		zip_safe=False,
		ext_modules=optimal_list,
		include_dirs=include_dirs
)
