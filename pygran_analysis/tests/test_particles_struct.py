import pygran_analysis as analysis
import sys


def test_system_rdf(trajf):
    # Create a granular object from a LIGGGHTS dump file
    sys = analysis.System(Particles=trajf, units="si")

    # Go to last frame
    sys.goto(-1)

    # Switch to micro unit system
    sys.units("micro")

    # Compute the radial distribution function
    g, r, _ = sys.Particles.computeRDF()

    assert r.max() > 4.0


def test_system_neighbors(trajf):

    # Create a granular object from a LIGGGHTS dump file
    sys = analysis.System(Particles=trajf, units="si")

    # Go to last frame
    sys.goto(-1)

    # Construct a class for nearest neighbor searching
    neigh = analysis.equilibrium.Neighbors(sys.Particles)

    # Extract coordination number per particle
    coon = neigh.coon
