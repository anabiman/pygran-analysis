import pygran_analysis as analysis
import sys
import numpy
from timer import timeit


def _foo(trajf):
    # Create a granular object from a LIGGGHTS dump file
    parts = analysis.Particles(fname=trajf)

    # Compute the radial distribution function
    kwargs = {"npts": 100, "optimize": True}
    time_op, result_op = timeit(parts.computeRDF, iters=1, **kwargs)
    g_op, r_op, _ = result_op
    print(g_op, r_op)

    # kwargs = {"npts": 100, "optimize": False}
    # time, result = timeit(parts.computeRDF, iters=200, **kwargs)
    # g, r, _ = result

    # print("RDF allclose:", numpy.allclose(g_op, g))
    # print("Radii allclose:", numpy.allclose(r_op, r))

    # print("Cython/Numpy:", time / time_op)


_foo("pygran_analysis/tests/data/particles1000.dump")
