import pygran_analysis as analysis
import pytest


def test_particles_basic(trajf):
    # Create a System object from a LIGGGHTS dump file. They keyword 'Particles' is mandatory, since it instructs
    # System to create an object of type 'Particles' which can be assessed from the instantiated System object.

    sys = analysis.System(Particles=trajf)

    # Go to last frame
    sys.goto(-1)
    assert sys.frame == 30

    # Create a reference to sys.Particles
    parts = sys.Particles

    # Any changes made to Particles is reflected in sys.Particles. To avoid that, a hard copy of Particles should be
    # created instead:
    parts = sys.Particles.copy()

    len_parts = len(parts)
    assert len_parts > 0

    nparts = 0
    # Looping over Particles yields a Particles object of length 1
    for part in parts:
        nparts += len(part)

    assert nparts == len_parts

    # Slice Particles into a new object containing the 1st 10 particles
    slice = parts[:10]
    assert len(slice) == 10

    # Slice Particles into a new object containing particles 1, 2, and 10
    slice = parts[[1, 2, 10]]
    assert len(slice) == 3

    # Slice Particles into a new class containing the 1st 10 particles and the last 10 particles
    slice = parts[:10] + parts[-10:]

    # More sophisticated slicing can be done with 1 or boolean expressions. For example:

    # Create a Particles object containing particles smaller than 25% of the mean particle diameter
    small_parts = parts[parts.radius <= parts.radius.mean() * 0.25]
    assert len(small_parts) == 0

    # Create a Particles object containing particles in the positive xy plane
    small_parts = parts[(parts.y >= 0) & (parts.x >= 0)]
    assert len(small_parts) > 0
