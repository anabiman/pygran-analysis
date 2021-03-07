import pygran_analysis as analysis
import numpy


def run_compute(Parts):
    Parts.computeRDF()
    Parts.computeMass(tdensity=1.0)
    Parts.computeROG()
    Parts.computeCOM()
    Parts.computeRadius()
    Parts.computeAngleRepose()

    # Parts.computeIntensitySegregation()
    # Parts.computeScaleSegregation()

    Parts.computeDensity(tdensity=1.0)
    Parts.computeDensityLocal(tdensity=1.0, dr=0.1, axis="x")
    Parts.computeVolume()


def test_constructor():
    natoms = 100
    x, y, z = (
        numpy.random.rand(natoms),
        numpy.random.rand(natoms),
        numpy.random.rand(natoms),
    )
    radii = numpy.ones(natoms) * 0.1
    Parts = analysis.Particles(
        data={"timestep": 0, "x": x, "y": y, "z": z, "natoms": natoms, "radius": radii},
        units="si",
    )
    run_compute(Parts)
