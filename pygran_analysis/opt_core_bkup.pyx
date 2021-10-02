"""
A submodule that provides the fundamental classes used in the analysis module

Created on July 10, 2016

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
and LICENSE files.
"""

try:
    import numpy
    cimport numpy
    from libc cimport math, stdlib
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"Cython must be installed to use module {__name__}.")

from scipy.stats import binned_statistic

ctypedef numpy.float DTYPE_f
ctypedef numpy.int DTYPE_i

#(numpy.ndarray[DTYPE_f, ndim=1], numpy.ndarray[DTYPE_f, ndim=1], numpy.ndarray[DTYPE_i, ndim=1]) 

cdef double mean(int m, int* a):
    cdef:
        int i
        double sum = 0
    for i in range(m):
        sum += a[i]
    return sum/m

cdef int gen_bins(
        double min, 
        double max,
        double incr,
        double** edges,
    ):

    cdef:
        int i
        int bin_count = <int> ((max - min) / incr) - 1
        
    for i in range(bin_count):
        edges[0][i] = min + (i+1) * incr

    return bin_count

cdef int which_bin(
        double data, 
        double bin_maxes[], 
        int bin_count, 
        double min_meas
    ):
   
    cdef:
        int bottom = 0
        int top = bin_count - 1
        int mid
        double bin_max
        double bin_min

    while bottom <= top:
        mid = (bottom + top) / 2
        bin_max = bin_maxes[mid]

        if mid:
            bin_min = bin_maxes[mid-1]
        else:
            bin_min = min_meas

        if data >= bin_max: 
            bottom = mid + 1
        elif data < bin_min:
            top = mid - 1
        else:
            return mid

cdef void histogram(
        double []data, 
        int data_count, 
        double []edges,
        int bin_count,
        double min_meas,
        int* bin_counts,
    ):

    cdef:
        int i
        int bin

    bin_counts = <int*> stdlib.calloc(bin_count, sizeof(int))

    for i in range(data_count):
        bin = which_bin(data[i], edges, bin_count, min_meas)
        bin_counts[bin] += 1
        #print("bin, count = ", bin, bin_counts[bin])

def computeRDF(numpy_x, numpy_y, numpy_z, dr, S, rMax):
    """Computes the three-dimensional radial distribution function for a set of
    spherical particles contained in a cube with side length S.  This simple
    function finds reference particles such that a sphere of radius rMax drawn
    around the particle will fit entirely within the cube, eliminating the need
    to compensate for edge effects.  If no such particles exist, an error is
    returned.  Try a smaller rMax...or write some code to handle edge effects! ;)

    :param dr: increment for increasing radius of spherical shell
    :type dr: float

    :param rMax: outer diameter of largest spherical shell
    :type rMax: float

    :return: (rdf as numpy array, radii of spherical shells as numpy array, indices of particles)
    :rtype: tuple
    """
    cdef: 
        int nparts = len(numpy_x)
        double max = rMax + 1.1 * dr
        double min = 0
        int num_increments = <int> ((max - min) / dr) - 1
        double *edges = <double*> stdlib.malloc(num_increments*sizeof(double))
        int *interior_indices = NULL
        double *g_average = NULL
        double *radii = NULL

    num_increments = gen_bins(min, max, dr, &edges)

    for i in range(num_increments):
        print(edges[i])
    
    print("Calling cython_computeRDF")
    cdef int nindices = cython_computeRDF(numpy_x, numpy_y, numpy_z, dr, S, rMax, nparts, num_increments, g_average, radii, interior_indices, edges)

    print("Converting pointers to numpy arrays")
    cdef numpy.npy_intp len_rdf = <numpy.npy_intp> num_increments, len_indices = <numpy.npy_intp> nindices

    print("num_increments, len_indices:", num_increments, len_indices)

    cdef numpy.ndarray[DTYPE_f, ndim=1] py_g_average = numpy.PyArray_SimpleNewFromData(1, &len_rdf, numpy.NPY_FLOAT, g_average)
    print("Done with g_av")
    cdef numpy.ndarray[DTYPE_f, ndim=1] py_radii = numpy.PyArray_SimpleNewFromData(1, &len_rdf, numpy.NPY_FLOAT, <void*>radii)
    cdef numpy.ndarray[DTYPE_i, ndim=1] py_interior_indices = numpy.PyArray_SimpleNewFromData(1, &len_indices, numpy.NPY_INT, <void*>interior_indices)

    print("Freeing memory")
    stdlib.free(edges)
    # stdlib.free(g_average)
    # stdlib.free(radii)
    # stdlib.free(interior_indices)

    return py_g_average, py_radii, py_interior_indices

cdef int cython_computeRDF(
        numpy.ndarray[double, ndim=1] numpy_x, 
        numpy.ndarray[double, ndim=1] numpy_y, 
        numpy.ndarray[double, ndim=1] numpy_z, 
        double dr, 
        double S, 
        double rMax, 
        int nparts,
        int num_increments,
        double[] g_average,
        double[] radii,
        int[] interior_indices,
        double[] edges
    ):
    cdef double[:] x = numpy.ascontiguousarray(numpy_x, dtype = numpy.float)
    cdef double[:] y = numpy.ascontiguousarray(numpy_y, dtype = numpy.float)
    cdef double[:] z = numpy.ascontiguousarray(numpy_z, dtype = numpy.float)
    cdef int i, j

    cdef int *bools1 = <int*> stdlib.malloc(nparts * sizeof(int))
    for i in range(nparts):
        bools1[i] = x[i] > rMax - S

    cdef int *bools2 = <int*> stdlib.malloc(nparts * sizeof(int))
    for i in range(nparts):
        bools2[i] = x[i] < S - rMax

    cdef int *bools3 = <int*> stdlib.malloc(nparts * sizeof(int))
    for i in range(nparts):
        bools3[i] = y[i] > rMax - S

    cdef int *bools4 = <int*> stdlib.malloc(nparts * sizeof(int))
    for i in range(nparts):
        bools4[i] = y[i] < S - rMax

    cdef int *bools5 = <int*> stdlib.malloc(nparts * sizeof(int)) 
    for i in range(nparts):
        bools5[i] = z[i] > rMax - S

    cdef int *bools6 = <int*> stdlib.malloc(nparts * sizeof(int))
    for i in range(nparts):
        bools6[i] = z[i] < S - rMax

    cdef int num_interior_particles, count = 0
    interior_indices = <int*> stdlib.malloc(nparts * sizeof(int))

    for i in range(nparts):
        insert = bools1[i] * bools2[i] * bools3[i] * bools4[i] * bools5[i] * bools6[i]
        if insert:
            interior_indices[count] = insert
            count += 1
    
    num_interior_particles = count
    if num_interior_particles < 1:
        raise RuntimeError(
            f"No particles found for which a sphere of radius rMax will lie entirely within a cube of side length {S}. "
            "Decrease rMax or increase the size of the cube."
        )

    cdef double **g = <double**> stdlib.malloc(num_interior_particles * sizeof(double*))
    for i in range(num_interior_particles):
        g[i] = <double*> stdlib.malloc(num_increments * sizeof(double))
 
    cdef double numberDensity = num_interior_particles / math.pow(S, 3.0)

    # Compute pairwise correlation for each interior particle
    cdef double *d = <double*> stdlib.malloc(nparts * sizeof(double))
    cdef int *result = <int*> stdlib.malloc(num_increments * sizeof(int))
    cdef int index, p

    for p in range(num_interior_particles):
        index = interior_indices[p]    
        for i in range(nparts):
            d[i] = math.sqrt( math.pow(x[index] - x[i], 2.0) + math.pow(y[index] - y[i], 2.0) + math.pow(z[index] - z[i], 2.0) )
        d[index] = 2.0 * rMax
        histogram(d, nparts, edges, num_increments, 0.0, result) # min_r = 0 -> gen_bins
        for i in range(num_increments):
            g[p][i] = result[i]

    g_average = <double*> stdlib.calloc(num_increments, sizeof(double))
    radii = <double*> stdlib.malloc(num_increments * sizeof(double))

    cdef double rOuter, rInner
    # Average g(r) for all interior particles and compute radii
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i + 1]) / 2.0
        rOuter = edges[i + 1]
        rInner = edges[i]
    
        for j in range(num_interior_particles):
            g_average[i] += g[j][i]
 
        g_average[i] /= ( 4.0 / 3.0 * math.pi * (rOuter ** 3 - rInner ** 3) * numberDensity * nparts)

    # Free allocated memory
    stdlib.free(bools1)
    stdlib.free(bools2)
    stdlib.free(bools3)
    stdlib.free(bools4)
    stdlib.free(bools5)
    stdlib.free(bools6)
    stdlib.free(result)
    stdlib.free(d)

    for i in range(num_interior_particles):
        stdlib.free(g[i])
    stdlib.free(g)

    # Number of particles in shell/total number of particles/volume of shell/number density
    # shell volume = 4/3*pi(r_outer**3-r_inner**3)
    return num_interior_particles

def computeIntensitySegregation(parts: "Particles", resol=None):
    """Computes the intensity of segregation for binary mixture
    as defined by Danckwerts:

    I = sigma_a**2 / (mean_a (1 - mean_a))

    :param resol: bin size for grid construction (default 3 * diameter)
    :type resol: float

    :return: intensity
    :rtype: float
    """

    if not resol:
        resol = parts.radius.min() * 3

    if hasattr(parts, "type"):
        ptypes = parts.type
    else:
        ptypes = numpy.ones(parts.natoms)

    if len(numpy.unique(ptypes)) != 2:
        raise ValueError(
            "Intensity of segergation can be computed only for a binary system."
        )
        # should I support tertiary systems or more ???

    nx = int((parts.x.max() - parts.x.min()) / resol) + 1
    ny = int((parts.y.max() - parts.y.min()) / resol) + 1
    nz = int((parts.z.max() - parts.z.min()) / resol) + 1

    indices = numpy.zeros((nx, ny, nz), dtype="float64")

    for sn, ctype in enumerate(numpy.unique(ptypes)):

        parts = parts[ptypes == ctype]

        parts.translate(
            value=(-parts.x.min(), -parts.y.min(), -parts.z.min()), attr=("x", "y", "z")
        )

        x = numpy.array(parts.x / resol, dtype="int64")
        y = numpy.array(parts.y / resol, dtype="int64")
        z = numpy.array(parts.z / resol, dtype="int64")

        for i in range(parts.natoms):
            indices[x[i], y[i], z[i]] += parts.radius[i] ** 3

        if sn == 0:
            indices_a = indices.copy()

    indices_a[indices > 0] /= indices[indices > 0]
    aMean = indices_a[indices > 0].mean()
    aStd = indices_a[indices > 0].std()

    return aStd ** 2 / (aMean * (1.0 - aMean)), indices_a, indices


def computeScaleSegregation(parts: "Particles", nTrials=1000, resol=None, npts=50, maxDist=None):
    """Computes the correlation coefficient as defined by Danckwerts:
    R(r) = a * b / std(a)**2

    This is done via a Monte Carlo simulation.

    :param resol: bin size for grid construction (default min radius)
    :param nTrials: number of Monte Carlo trials (sample size)
    :param npts: number of bins for histogram construction
    @[maxDist]: maximum distance (in units of grid size) to sample

    Returns the coefficient of correlation R(r) and separation distance (r)
    """

    if not resol:
        resol = parts.radius.min()

    _, a, total = parts.computeIntensitySegregation(resol)

    if not maxDist:
        maxDim = max(a.shape)
        maxDist = int(numpy.sqrt(3 * maxDim ** 2)) + 1

    volMean = a[total > 0].mean()
    volVar = a[total > 0].std() ** 2

    corr = numpy.zeros(nTrials)
    dist = numpy.zeros(nTrials)
    count = 0

    # Begin Monte Carlo simulation
    while count < nTrials:

        i1, i2 = numpy.random.randint(0, a.shape[0], size=2)
        j1, j2 = numpy.random.randint(0, a.shape[1], size=2)
        k1, k2 = numpy.random.randint(0, a.shape[2], size=2)

        # Make sure we are sampling non-void spatial points
        if total[i1, j1, k1] > 0 and total[i2, j2, k2] > 0:

            distance = numpy.sqrt((i2 - i1) ** 2 + (j2 - j1) ** 2 + (k2 - k1) ** 2)

            if distance <= maxDist:

                corr[count] = (
                    (a[i1, j1, k1] - volMean) * (a[i2, j2, k2] - volMean)
                ) / volVar
                dist[count] = distance
                count += 1

    corrfunc, distance, _ = binned_statistic(dist, corr, "mean", npts)

    return (
        corrfunc[numpy.invert(numpy.isnan(corrfunc))],
        distance[0:-1][numpy.invert(numpy.isnan(corrfunc))] * resol,
    )
