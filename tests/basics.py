import PyGranAnalysis as analysis
import numpy as np


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
    x, y, z = np.random.rand(natoms), np.random.rand(natoms), np.random.rand(natoms)
    radii = np.ones(natoms) * 0.1
    Parts = analysis.Particles(
        data={"timestep": 0, "x": x, "y": y, "z": z, "natoms": natoms, "radius": radii},
        units="si",
    )
    run_compute(Parts)


test_constructor()
