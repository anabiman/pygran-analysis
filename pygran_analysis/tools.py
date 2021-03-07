"""
Created on Oct 13, 2019
Author: Andrew Abi-Mansour

This is the::
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
"""

import os

# Conversion unit systems
# si.: meters, seconds, kilograms
# cgs: centimeters, seconds, grams
# micro: microns, seconds, micrograms
# nano: nanometers, seconds, nanograms

conversion = {
    "si": {"distance": [1, "m"], "time": [1, "s"], "mass": [1, "kg"]},
    "cgs": {"distance": [1e-2, "cm"], "time": [1, "s"], "mass": [1e-3, "g"]},
    "micro": {
        "distance": [1e-6, "$\mu m$"],
        "time": [1, "s"],
        "mass": [1e-9, "$\mu g$"],
    },
    "nano": {"distance": [1e-9, "nm"], "time": [1, "s"], "mass": [1e-12, "ng"]},
}


def convert(unitso, unitsf):
    """Generic function that converts length/time/mass from one unit system to another

    :param unitso: unit system to convert from
    :type unitso: str

    :param unitsf: unit system to convert to
    :type unitsf: str

    :return: new unit system
    :rtype: dict
    """

    if unitso in conversion:
        if unitsf in conversion:
            conv = conversion[unitso]

            for key in conv:
                conv[key][0] /= conversion[unitsf][key][0]

            return conv
        else:
            raise ValueError("Input unit system not supported: {}".format(unitsf))
    else:
        raise ValueError("Input unit system not supported: {}".format(unitso))


def find(fname, path):
    """Finds a filename (fname) along the path `path'

    :param fname: filename
    :type fname: str

    :param path: search path
    :type path: str

    :return: absolute path of the fname if found, else None
    :rtype: str/None
    """
    for root, dirs, files in os.walk(path):
        if fname in files:
            return os.path.join(root, fname)

    return None


def run(program):
    """Unix only: launches an executable program available in the PATH environment variable.

    :param program: name of the executable to search for
    :type program: str

    :return: 0 if successful and 1 otherwise
    :rtype: bool
    """
    paths = os.environ["PATH"]

    for path in paths.split(":"):
        found = Tools.find(program, path)

        if found:
            print("Launching {}".format(found))
            os.system(found + " &")
            return 0

    print("Could not find {} in {}".format(program, paths))
    return 1
