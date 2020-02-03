cimport numpy as np
import numpy as np

from libc.math cimport log, exp

from numpy.linalg import inv
import math
# from scipy.stats import multivariate_normal
import cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)

cpdef double[:] c_convert_into_probabilities (double[:] log_proba, double[:] norm_probas ):
    """
    Computes normalized probabilities. Takes as input a 1*n vector
    of unnormalized probabilities. Log-Sum-Exp's them to avoid
    overflows and returns a 1*n vector of probabilities.

    Output
    - None

    Argument:
    - `log_proba` a 1*n array of of unnormalized log-probabilities.
    - norm_probas qui doit être du type np.empty(n, dtype = np.double) où est n est le size de log_proba
    """

    # Get the size of the vector
    cdef int n = log_proba.size
#    cdef size_t n = sizeof(log_proba) / sizeof(*log_proba)

    # Get the max of the vector
    cdef double max_log = 0.0

    for i from 0 <= i < n:
        if log_proba[i] > max_log :
            max_log = log_proba[i]

    # Remove the max to each element of the vector
    cdef double[:] clean_proba = np.empty(n, dtype=np.double)

    for i from 0 <= i < n:
        clean_proba[i] = log_proba[i] - max_log

    # Take the exponential to get linear values
    cdef double[:] probas = np.empty(n, dtype=np.double)

    for i from 0 <= i < n:
        probas[i] = exp(log_proba[i])

    # Compute the normalizing constant
    cdef float total_proba = 0.0

    for i from 0 <= i < n:
        total_proba = total_proba + probas[i]


    for i from 0 <= i < n:
        norm_probas[i] = probas[i] / total_proba


    return None

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef c_compute_log_probability (double[:] coordinate, double[:,:] tau_0, double[:] mu_0 , double[:,:] tau_x, double[:,:] Sigma_x, int obs_in_cluster, double[:] sum_data):

    """
    Computes the unormalized log probability of belonging the ith cluster. The probability corresponds to
    the value of a gaussian of mean mu_p and covariance sigma_p evaluated at point `coordinates`

    Returns None

    Arguments:

    - `res` : does the computations on this guy

    """
    cdef double res

    cdef int n = tau_0.shape[0] # Vectors are assumed to have the same length
                                # this condition is necessarily satisfied, otherwhise
                                # the whole algorithm wouldn't work
                                # Furthermore, matrices are SDP matrices.
    cdef int m = tau_0.shape[1]


    # Compute tau_p the cluster specific precision

    # Compute the scalar/vector sum
    cdef double[:,:] scaled_matrix = np.empty((n,m), dtype=np.double)

    for i from 0 <= i < n:
        for j from 0 <= j < m:
            scaled_matrix[i,j] = tau_x[i,j] * obs_in_cluster

    # Compute the vector/vector sum
    cdef double[:,:] tau_p = np.empty((n,m), dtype=np.double)

    for i from 0 <= i < n:
        for j from 0 <= j < m:
            tau_p[i,j] = tau_0[i,j] + scaled_matrix[i,j]


    Sigma_p = inv(tau_p)


    # Compute mu_p the cluster specific mean

    #mean_p = np.dot(Sigma_p , (np.dot(tau_x,sum_data) + np.dot(tau_0,mu_0)))

    # Product tau_0 * mu_0
    cdef double[:] tau_mu_0  = np.empty(n, dtype = np.double)
    cdef double[:] tau_data = np.empty(n, dtype = np.double)
    cdef double[:] mean_p = np.empty(n, dtype = np.double)
    cdef double[:] data_mu_0 = np.empty(n, dtype = np.double)
    cdef double[:,:] tau_xp = np.empty((tau_x.shape[0],tau_x.shape[1]), dtype = np.double)


#    matrix_multiply1(tau_0, mu_0, tau_mu_0)
    tau_mu_0 = np.dot(tau_0, mu_0)

#    matrix_multiply1(tau_x, sum_data, tau_data)
    tau_data = np.dot(tau_x, sum_data)


    # Do the addition

    vector_addition1(tau_mu_0, tau_data, data_mu_0)

    # do the final product

#    matrix_multiply1(Sigma_p,data_mu_0,mean_p)
    mean_p = np.dot(Sigma_p, data_mu_0)

    # Addition des matrices

    matrix_addition1(tau_p, tau_x, tau_xp)



    # retourne

    # res = math.log(obs_in_cluster) + multivariate_normal.logpdf(coordinate, mean= mean_p, cov = Sigma_xp)
    res = math.log(obs_in_cluster) + evaluate_log_unnormalized_gaussian(coordinate, mean_p, tau_xp)


    return res

cpdef c_compute_log_probability_not_cluster (double[:] coordinate, double alpha, double[:] mu_0 , double[:,:] tau_x, double[:,:] tau_0):
    """
    Computes the unormalized log probaility not to beloging to any cluster.

    Returns res

    Arguments:


    """

    cdef double res

    # Somme des deux matrics
    cdef double[:,:] tau_x0 = np.empty((tau_x.shape[0],tau_x.shape[1]), dtype = np.double)
    matrix_addition1(tau_0, tau_x, tau_x0)

    # Calcule de la valeur
    #res = math.log(alpha) + multivariate_normal.logpdf(coordinate, mean= mu_0, cov = Sigma_x0)
    res = math.log(alpha) + evaluate_log_unnormalized_gaussian(coordinate, mu_0, tau_x0)

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
#cpdef matrix_soustraction1(double[:,:] u, double[:,:] v, double[:,:] res):
#    """
#    Soustraction de matrice U - V
#    """
#
#    assert (v.shape[0] == u.shape[0]) and (u.shape[1] == v.shape[1]), "Dimension mismatch."
#    cdef int i, j, k
#    cdef int m, n
#
#    m = u.shape[0]
#    n = u.shape[1]
#
#    with cython.nogil:
#        for i in range(n):
#            for j in range(m):
#                res[i,j] += u[i,j] - v[i,j]
#    return None


@cython.boundscheck(False)
@cython.wraparound(False)
#cpdef vector_soustraction1(double[:] u, double[:] v, double[:] res):
#    """
#    Soustraction de vecteur U - V
#    """
#
#    assert (v.size == u.size), "Dimension mismatch."
#
#    cdef int i, j, k
#    cdef int m
#
#    m = u.size
#
#    with cython.nogil:
#        for i in range(m):
#            res[i] += u[i] - v[i]
#
#    return None
#
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef matrix_addition1(double[:,:] u, double[:,:] v, double[:,:] res):

    """
    Addition de matrice U et V
    """

    assert (v.shape[0] == u.shape[0]) and (u.shape[1] == v.shape[1]), "Dimension mismatch."

    cdef int i, j, k
    cdef int m, n

    m = u.shape[0]
    n = u.shape[1]

    with cython.nogil:
        for i in range(n):
            for j in range(m):
                res[i,j] += v[i,j] + u[i,j]
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vector_addition1(double[:] u, double[:] v, double[:] res):

    """
    Addition de vecteur U et V
    """

    assert (v.size == u.size), "Dimension mismatch."
    cdef int i, j, k
    cdef int m

    m = u.size

    with cython.nogil:
        for i in range(m):
            res[i] += u[i] + v[i]

    return None

@cython.boundscheck(False)
cpdef double evaluate_log_unnormalized_gaussian (double[:] coordinates, double[:] mean, double[:,:] inv_covariance):
    """
    computes the log value of a gaussian (without the constant) at
    point coordinates.

    returns a float corresponding to that value.

    arguments:

    `coordinates` : a vector corresponding to the coordinates of the observations from which
    the density is evaluated

    `mean` : a vector corresponding to the mean of the gaussian distribution

    `inv_covariance` the inverse variance covariance matrix of the gaussian distribution
    """

    # get the dimension of the coordinate
    cdef int n = coordinates.size

    # set up parameters
    cdef double res
    cdef double[:] x_minus_mu = np.empty(n, dtype = np.double)
    cdef double[:] xA = np.empty(n, dtype = np.double)
    cdef double inv_det

    # compuation is done as follows : first compute the difference between two vectors
    # coordinates and mean, then compute the product and return the value

    # compute the difference x_minus_mu
    with cython.nogil:
        for i in range(n):
            x_minus_mu[i] = coordinates[i] - mean[i]

    # compute the first part of the product, xA
    with cython.nogil:
        for i in range(n):
            for j in range(n):
                xA[j] += x_minus_mu[i] * inv_covariance[i,j]

    # compute the second part of the product
    with cython.nogil:
        for i in range(n):
            res += xA[i] * x_minus_mu[i]


    # compute the determinant of the matrix
    inv_det = np.linalg.det(inv_covariance)

    return - 1 / 2 * ( n * log(2 * math.pi) + 1 / inv_det + res)


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef matrix_multiply1(double[:,:] u, double[:, :] v, double[:, :] res):
#
#    """
#
#    Multiplication de matrice U et V
#
#    """
#
#    cdef int i, j, k
#    cdef int m, n, p
#
#    m = u.shape[0]
#    n = u.shape[1]
#    p = v.shape[1]
#
#    with cython.nogil:
#        for i in range(m):
#            for j in range(p):
#                res[i,j] = 0
#                for k in range(n):
#                    res[i,j] += u[i,k] * v[k,j]
#
#    return None
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef scalarmatrix_multiply(double lamb, double[:, :] u, double[:, :] res):
#
#    """
#    Multiplication d'une matrice avec un scalaire
#
#    """
#
#    cdef int i, j, k
#    cdef int m, n, p
#
#    m = u.shape[0]
#    n = u.shape[1]
#
#    with cython.nogil:
#        for i in range(m):
#            for j in range(n):
#                res[i,j] = 0
#                res[i,j] += lamb*u[i,j]
#
#    return None
