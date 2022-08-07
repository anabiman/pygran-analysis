import pytest

import pygran_analysis as analysis


def test_system_parts(trajf):
    # Create a System object from a LIGGGHTS dump file. They keyword 'Particles' is mandatory, since it instructs
    # System to create an object of type 'Particles' which can be assessed from the instantiated System object.
    sys = analysis.System(Particles=trajf)
    assert sys.frame == 0
    sys.skip()  # skip any empty frames
    assert sys.frame == 1
    sys.next()
    assert sys.frame == 2
    sys.goto(-1)  # go to last frame
    assert len(sys) == 1


@pytest.mark.skipif(not analysis.core.vtk, reason="VTK not available")
def test_system_parts_mesh(trajf, meshf):
    # Create a System object from a LIGGGHTS dump file. They keyword 'Particles' is mandatory, since it instructs
    # System to create an object of type 'Particles' which can be assessed from the instantiated System object.
    sys = analysis.System(Particles=trajf, Mesh=meshf)
    assert sys.frame == 0
    sys.skip()  # skip any empty frames
    assert sys.frame == 0
    sys.next()
    assert sys.frame == 1
    sys.goto(-1)  # go to last frame
    assert len(sys) == 2
