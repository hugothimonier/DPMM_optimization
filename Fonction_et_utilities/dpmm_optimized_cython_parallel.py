# coding: utf-8

# Library imports
import cython
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import invwishart
import sys
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from progressbar import ProgressBar
import multiprocessing
import time
from sklearn.utils import shuffle
import collections
import cython
import dpmm_cython_functions


def cython_dpmm_algorithm_(data, args, savetime = False, traceback = False):
    """

    Implementation of a DP-GMM algorithm for cluster assignation.

    Returns a n*1 vector with the clusters labels assignated to each observation
    If traceback = True, then returns the whole matrix where each column corresponds
    to an interation of clusters assignation.

    Arguments:

        `data` a DataFrame (nxd) containing the observations. n corresponds to the number of
        observations and d the number of columns to the dimension of the observations

        `args` a signed tuple of parameters. These parameters feature :
        `alpha` the prior on the GEM distribution (float)
        `mu_0`, the prior mean of the clusters' distribution (d*1 array)
        `Sigma_0` the prior variance of the clusters' distribution (d*d array)
        `sigma_x` the noise on the observations (float)
        `n_iter` the number of iterations (integer)
        `c_init` the initial clusters assignation (n * 1 array)

        `savetime` : a boolean indicating if the user wants the total execution time to be
        returned.

        `traceback` : a boolean indicating whether the users wants the full matrix of results
        to be returned.
    """


    # Step 1. Initialization of the parameters.


    # Set up the timer

    if savetime :
        start_time = time.time()

    # Unwrap the parameters
    alpha, mu_0, Sigma_0, sigma_x, n_iter, c_init = args.alpha, args.mu_0, args.Sigma_0, args.sigma_x, args.n_iter, args.c_init

    # Convert the DataFrame into an array
    data = data.values

    # Get the dimension of the data
    n_obs, data_dim = data.shape

    # Compute the precision parameters
    tau_0 = inv(Sigma_0)

    # Precision of the observations (scaled up) and associated precision
    Sigma_x = sigma_x * np.eye(data_dim)
    tau_x = 1/sigma_x * np.eye(data_dim)


    # Define an InnerParameters tuple. This tuple stores the parameters that will be used for computing the
    # log probabilities of belonging to the clusters.
    InnerParameters = collections.namedtuple('InnerParameters', ['mu_0','Sigma_0', 'tau_0', 'Sigma_x', 'tau_x', 'alpha'])
    p = InnerParameters(mu_0 = mu_0, Sigma_0 = Sigma_0, tau_0 = tau_0, Sigma_x = Sigma_x, tau_x = tau_x, alpha = alpha)

    # Initial cluster assignment
    z = c_init.astype(int)

    # Counts per cluster : dictionnary with key = label, value = count
    unique, counts = np.unique(z, return_counts = True)
    n_k = dict(zip(unique, counts))

    # Initial number of clusters
    n_clust = len(n_k.keys())

    # Initialization of the matrix of cluster membership.
    if traceback :
        # In this case, we will populate a matrix of n_obs * i_iter elements
        res =  np.empty((n_obs, n_iter))


    # Initialize a progress bar (function quite long to run)
    #pbar = ProgressBar()

    # Step 2. Main loop

    #for iter in pbar(range(n_iter)):
    for iter in range(n_iter):

        for n in range(n_obs):


            # Get the cluster of the nth obs
            c_i = z[n]

            # Remove the observation from the cluster count
            n_k[c_i] -= 1

            # If there is nobody left in this cluster, remove it
            # and shift the remaining clusters
            if n_k[c_i] == 0:
                # Put the number of observations of the last cluster into the
                # now empty one
                n_k[c_i] = n_k[n_clust]

                # Reassign the labels of the observations of the now empty
                # last cluster
                loc_z = np.where(z == n_clust)
                z[loc_z] = c_i

                # Remove the last cluster
                del n_k[n_clust]

                # Decrease the number of clusters
                n_clust -= 1

            # Make sure the current observation will not be counted
            # as a cluster
            z[n] = -1

            # Define the vector of log probabilities
            logp = np.empty(n_clust + 1)

            # Get the coordinates of the current observation
            coordinates = data[n,:]

            # Compute the unnormalized log probability of belonging
            # to the ith cluster.
            for c_i in n_k.keys():

                # Get the number of observations in that cluster
                obs_in_cluster = n_k[c_i]

                # Find the points belonging to that cluster
                loc_z = np.where(z == c_i)

                # Sum the points of this cluster.
                sum_data = np.sum(data[loc_z], axis = 0)

                # Compute the log probability of belonging to cluster c_i
                res_1 = dpmm_cython_functions.c_compute_log_probability(coordinates, np.asarray(tau_0, dtype = np.double), np.asarray(mu_0, dtype = np.double) , np.asarray(tau_x, dtype = np.double), np.asarray(Sigma_x, dtype = np.double), obs_in_cluster, np.asarray(sum_data, dtype = np.double))
                logp[c_i-1] = res_1

            # Compute the probability not to belong to any cluster

            res_2 = dpmm_cython_functions.c_compute_log_probability_not_cluster(coordinates, np.double(alpha), np.asarray(mu_0, dtype = np.double) , np.asarray(Sigma_x, dtype = np.double), np.asarray(Sigma_0, dtype = np.double))
            logp[n_clust] = res_2

            # Convert into probabilities
            loc_probs = np.empty(logp.size, dtype = np.double)
            dpmm_cython_functions.c_convert_into_probabilities(np.asarray(logp, dtype = np.double), loc_probs)

            # Given these probabilities, sample a cluster assignation
            # to the observation
            loc_probs = np.array([0.01 if math.isnan(x) else x for x in loc_probs])
            loc_probs /= loc_probs.sum().astype(float)

            newz = int(np.random.choice(n_clust+1, 1, p = loc_probs))
            newz += 1   # increase to translate the outcomte into a cluster label

            # Spawn a new cluster if necessary
            if newz == n_clust + 1:
                n_k[newz] = 0 # initialize a new key in the dictionnary.
                n_clust += 1 # increase the number of clusters.

            z[n] = newz # assign the new cluster value to the obs.
            n_k[newz] += 1 # increase the number of observation in the cluster.

        # Write the output of the current iteration in the output matrix
        # if traceback = True
        if traceback :
            res[:,iter] = z


    # Step 3. Outputs

    if not traceback :
        # The results are simply the last version in memory of z
        res = z

    if savetime :
        elapsed_time = time.time() - start_time
        return res, elapsed_time
    else:
        return res

def cython_dpmm_algorithm1(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_init, savetime = False, traceback = False):
    """

    Implementation of a DP-GMM algorithm for cluster assignation.

    Returns a n*1 vector with the clusters labels assignated to each observation
    If traceback = True, then returns the whole matrix where each column corresponds
    to an interation of clusters assignation.

    Arguments:

        `data` a DataFrame (nxd) containing the observations. n corresponds to the number of
        observations and d the number of columns to the dimension of the observations

        `args` a signed tuple of parameters. These parameters feature :
        `alpha` the prior on the GEM distribution (float)
        `mu_0`, the prior mean of the clusters' distribution (d*1 array)
        `Sigma_0` the prior variance of the clusters' distribution (d*d array)
        `sigma_x` the noise on the observations (float)
        `n_iter` the number of iterations (integer)
        `c_init` the initial clusters assignation (n * 1 array)

        `savetime` : a boolean indicating if the user wants the total execution time to be
        returned.

        `traceback` : a boolean indicating whether the users wants the full matrix of results
        to be returned.
    """


    # Step 1. Initialization of the parameters.


    # Set up the timer

    if savetime :
        start_time = time.time()

    # Convert the DataFrame into an array
    data = data.values

    # Get the dimension of the data
    n_obs, data_dim = data.shape

    # Compute the precision parameters
    tau_0 = inv(Sigma_0)

    # Precision of the observations (scaled up) and associated precision
    Sigma_x = sigma_x * np.eye(data_dim)
    tau_x = 1/sigma_x * np.eye(data_dim)


    # Define an InnerParameters tuple. This tuple stores the parameters that will be used for computing the
    # log probabilities of belonging to the clusters.
    InnerParameters = collections.namedtuple('InnerParameters', ['mu_0','Sigma_0', 'tau_0', 'Sigma_x', 'tau_x', 'alpha'])
    p = InnerParameters(mu_0 = mu_0, Sigma_0 = Sigma_0, tau_0 = tau_0, Sigma_x = Sigma_x, tau_x = tau_x, alpha = alpha)

    # Initial cluster assignment
    z = c_init.astype(int)

    # Counts per cluster : dictionnary with key = label, value = count
    unique, counts = np.unique(z, return_counts = True)
    n_k = dict(zip(unique, counts))

    # Initial number of clusters
    n_clust = len(n_k.keys())

    # Initialization of the matrix of cluster membership.
    if traceback :
        # In this case, we will populate a matrix of n_obs * i_iter elements
        res =  np.empty((n_obs, n_iter))


    # Initialize a progress bar (function quite long to run)
    # pbar = ProgressBar()

    # Step 2. Main loop

    #for iter in pbar(range(n_iter)):
    for iter in range(n_iter):

        for n in range(n_obs):


            # Get the cluster of the nth obs
            c_i = z[n]

            # Remove the observation from the cluster count
            n_k[c_i] -= 1

            # If there is nobody left in this cluster, remove it
            # and shift the remaining clusters
            if n_k[c_i] == 0:
                # Put the number of observations of the last cluster into the
                # now empty one
                n_k[c_i] = n_k[n_clust]

                # Reassign the labels of the observations of the now empty
                # last cluster
                loc_z = np.where(z == n_clust)
                z[loc_z] = c_i

                # Remove the last cluster
                del n_k[n_clust]

                # Decrease the number of clusters
                n_clust -= 1

            # Make sure the current observation will not be counted
            # as a cluster
            z[n] = -1

            # Define the vector of log probabilities
            logp = np.empty(n_clust + 1)

            # Get the coordinates of the current observation
            coordinates = data[n,:]

            # Compute the unnormalized log probability of belonging
            # to the ith cluster.
            for c_i in n_k.keys():

                # Get the number of observations in that cluster
                obs_in_cluster = n_k[c_i]

                # Find the points belonging to that cluster
                loc_z = np.where(z == c_i)

                # Sum the points of this cluster.
                sum_data = np.sum(data[loc_z], axis = 0)

                # Compute the log probability of belonging to cluster c_i
                res_1 = dpmm_cython_functions.c_compute_log_probability(coordinates, np.asarray(tau_0, dtype = np.double), np.asarray(mu_0, dtype = np.double) , np.asarray(tau_x, dtype = np.double), np.asarray(Sigma_x, dtype = np.double), obs_in_cluster, np.asarray(sum_data, dtype = np.double))
                logp[c_i-1] = res_1

            # Compute the probability not to belong to any cluster

            res_2 = dpmm_cython_functions.c_compute_log_probability_not_cluster(coordinates, np.double(alpha), np.asarray(mu_0, dtype = np.double) , np.asarray(Sigma_x, dtype = np.double), np.asarray(Sigma_0, dtype = np.double))
            logp[n_clust] = res_2

            # Convert into probabilities
            loc_probs = np.empty(logp.size, dtype = np.double)
            dpmm_cython_functions.c_convert_into_probabilities(np.asarray(logp, dtype = np.double), loc_probs)

            # Given these probabilities, sample a cluster assignation
            # to the observation
            loc_probs = np.array([0.01 if math.isnan(x) else x for x in loc_probs])
            loc_probs /= loc_probs.sum().astype(float)

            newz = int(np.random.choice(n_clust+1, 1, p = loc_probs))
            newz += 1   # increase to translate the outcomte into a cluster label

            # Spawn a new cluster if necessary
            if newz == n_clust + 1:
                n_k[newz] = 0 # initialize a new key in the dictionnary.
                n_clust += 1 # increase the number of clusters.

            z[n] = newz # assign the new cluster value to the obs.
            n_k[newz] += 1 # increase the number of observation in the cluster.

        # Write the output of the current iteration in the output matrix
        # if traceback = True
        if traceback :
            res[:,iter] = z


    # Step 3. Outputs

    if not traceback :
        # The results are simply the last version in memory of z
        res = z

    if savetime :
        elapsed_time = time.time() - start_time
        return res, elapsed_time
    else:
        return res

def stock_prob(clusters, logprob, z, data, tau_0, mu_0, tau_x, Sigma_x, coordinates):
    """
    Calcule la proba d'appartenir un cluster et la stock dans le vecteur logp
    Utilisé dans l'algo dpmm pas d'utilisation externe

    Arguments :
    clusters : dictionnaire qui contient le label du cluster et les observations pour chacun des cluster
    logprob : array vide de taille nombre de cluster + 1

    """

    for c_i in clusters.keys():

        # Get the number of observations in that cluster
        obs_in_cluster = clusters[c_i]

        # Find the points belonging to that cluster
        loc_z = np.where(z == c_i)

        # Sum the points of this cluster.
        sum_data = np.sum(data[loc_z], axis = 0)

        # Compute the log probability of belonging to cluster c_i
        res_1 = dpmm_cython_functions.c_compute_log_probability(coordinates, np.asarray(tau_0, dtype = np.double), np.asarray(mu_0, dtype = np.double) , np.asarray(tau_x, dtype = np.double), np.asarray(Sigma_x, dtype = np.double), obs_in_cluster, np.asarray(sum_data, dtype = np.double))
        logprob[c_i-1] = res_1

        return None


def cython_dpmm_algorithm2(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_init, savetime = False, traceback = False):

    """

    Implementation of a DP-GMM algorithm for cluster assignation.

    Returns a n*1 vector with the clusters labels assignated to each observation
    If traceback = True, then returns the whole matrix where each column corresponds
    to an interation of clusters assignation.

    Arguments:

        `data` a DataFrame (nxd) containing the observations. n corresponds to the number of
        observations and d the number of columns to the dimension of the observations

        `args` a signed tuple of parameters. These parameters feature :
        `alpha` the prior on the GEM distribution (float)
        `mu_0`, the prior mean of the clusters' distribution (d*1 array)
        `Sigma_0` the prior variance of the clusters' distribution (d*d array)
        `sigma_x` the noise on the observations (float)
        `n_iter` the number of iterations (integer)
        `c_init` the initial clusters assignation (n * 1 array)

        `savetime` : a boolean indicating if the user wants the total execution time to be
        returned.

        `traceback` : a boolean indicating whether the users wants the full matrix of results
        to be returned.
    """


    # Step 1. Initialization of the parameters.


    # Set up the timer

    if savetime :
        start_time = time.time()

    # Unwrap the parameters
    #alpha, mu_0, Sigma_0, sigma_x, n_iter, c_init = args.alpha, args.mu_0, args.Sigma_0, args.sigma_x, args.n_iter, args.c_init

    # Convert the DataFrame into an array
    data = data.values

    # Get the dimension of the data
    n_obs, data_dim = data.shape

    # Compute the precision parameters
    tau_0 = inv(Sigma_0)

    # Precision of the observations (scaled up) and associated precision
    Sigma_x = sigma_x * np.eye(data_dim)
    tau_x = 1/sigma_x * np.eye(data_dim)


    # Define an InnerParameters tuple. This tuple stores the parameters that will be used for computing the
    # log probabilities of belonging to the clusters.
    InnerParameters = collections.namedtuple('InnerParameters', ['mu_0','Sigma_0', 'tau_0', 'Sigma_x', 'tau_x', 'alpha'])
    p = InnerParameters(mu_0 = mu_0, Sigma_0 = Sigma_0, tau_0 = tau_0, Sigma_x = Sigma_x, tau_x = tau_x, alpha = alpha)

    # Initial cluster assignment
    z = c_init.astype(int)

    # Counts per cluster : dictionnary with key = label, value = count
    unique, counts = np.unique(z, return_counts = True)
    n_k = dict(zip(unique, counts))

    # Initial number of clusters
    n_clust = len(n_k.keys())

    # Initialization of the matrix of cluster membership.
    if traceback :
        # In this case, we will populate a matrix of n_obs * i_iter elements
        res =  np.empty((n_obs, n_iter))


    # Initialize a progress bar (function quite long to run)
    # pbar = ProgressBar()

    # Step 2. Main loop

    # for iter in pbar(range(n_iter)):
    for iter in range(n_iter):

        for n in range(n_obs):


            # Get the cluster of the nth obs
            c_i = z[n]

            # Remove the observation from the cluster count
            n_k[c_i] -= 1

            # If there is nobody left in this cluster, remove it
            # and shift the remaining clusters
            if n_k[c_i] == 0:
                # Put the number of observations of the last cluster into the
                # now empty one
                n_k[c_i] = n_k[n_clust]

                # Reassign the labels of the observations of the now empty
                # last cluster
                loc_z = np.where(z == n_clust)
                z[loc_z] = c_i

                # Remove the last cluster
                del n_k[n_clust]

                # Decrease the number of clusters
                n_clust -= 1

            # Make sure the current observation will not be counted
            # as a cluster
            z[n] = -1

            # Define the vector of log probabilities
            logp = np.empty(n_clust + 1)

            # Get the coordinates of the current observation
            coordinates = data[n,:]

            # Compute the unnormalized log probability of belonging
            # to the ith cluster.


            # Parallélisation

            n_k1 = dict(list(n_k.items())[len(n_k)//2:])
            n_k2 = dict(list(n_k.items())[:len(n_k)//2])

            logp1 = np.empty(len(n_k1.keys()))
            logp2 = np.empty(len(n_k2.keys())+1)


            args_ = [(n_k1, logp1, z, data, tau_0, mu_0, tau_x, Sigma_x, coordinates),
                    (n_k2, logp2, z, data, tau_0, mu_0, tau_x, Sigma_x, coordinates)]

            if __name__ == '__main__':
                with multiprocessing.Pool(processes=2) as p:
                    logp = p.starmap(stock_prob, args_)

            # Compute the probability not to belong to any cluster

            res_2 = dpmm_cython_functions.c_compute_log_probability_not_cluster(coordinates, np.double(alpha), np.asarray(mu_0, dtype = np.double) , np.asarray(Sigma_x, dtype = np.double), np.asarray(Sigma_0, dtype = np.double))
            logp[n_clust] = res_2

            # Convert into probabilities
            loc_probs = np.empty(logp.size, dtype = np.double)
            dpmm_cython_functions.c_convert_into_probabilities(np.asarray(logp, dtype = np.double), loc_probs)

            # Given these probabilities, sample a cluster assignation
            # to the observation
            loc_probs = np.array([0.01 if math.isnan(x) else x for x in loc_probs])
            loc_probs /= loc_probs.sum().astype(float)

            newz = int(np.random.choice(n_clust+1, 1, p = loc_probs))
            newz += 1   # increase to translate the outcomte into a cluster label

            # Spawn a new cluster if necessary
            if newz == n_clust + 1:
                n_k[newz] = 0 # initialize a new key in the dictionnary.
                n_clust += 1 # increase the number of clusters.

            z[n] = newz # assign the new cluster value to the obs.
            n_k[newz] += 1 # increase the number of observation in the cluster.

        # Write the output of the current iteration in the output matrix
        # if traceback = True
        if traceback :
            res[:,iter] = z


    # Step 3. Outputs

    if not traceback :
        # The results are simply the last version in memory of z
        res = z

    if savetime :
        elapsed_time = time.time() - start_time
        return res, elapsed_time
    else:
        return res


def mean_time(function, n_iter2, args):

    """

    Calcule le temps moyen d'execution d'une fonction

    Arguments :

    `function` : fonction à évaluer
    `n_iter2` : nombre d'itérations sur lesquelles est calculée la moyenne
    `args` : les paramètres de la fonction évaluée, ce doit être soit la forme d'un tuple
            par exemple pour la fonction dpmm : (data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled)

    """

    time = []

    for i in range(n_iter2) :

        output, time_ = function(*args, True)
        time.append(time_)

    del output, time_

    mean = np.mean(time)

    return mean




def cython_dpmm_algorithm_multiprocessing(data, numdatasplit, alpha, mu_0, Sigma_0, sigma_x, n_iter, savetime = False, __name__ = '__main__'):

    '''
    Fonction cythonisée de l'algorithme DPMM avec comme clusters de départ des cluster pré-définis par un autre DPMM
    sur des données scindées en numdatasplit sous données.

    Arguments :

    `data` : np.array type dataframe
    `numdatasplit` : nombre de split des données
    `alpha`, `mu_0`, `Sigma_0`, `sigma_x` : arguments pour l'algorithme de DPMM
    `n_iter` : nombre d'itération pour l'algorithme de Gibbs
    `savetime` : boolean to determine wether to output time
    '''

    assert numdatasplit in [2,4], 'Le nombre de datasplit doit être 2 ou 4'
    global results

    assert savetime in [True, False]

    if savetime :
        start_time = time.time()

    data = shuffle(data)

    if numdatasplit == 2 :
        data1, data2 = np.array_split(data.sample(frac=1), 2)
        c_pooled_1 = np.ones(data1.shape[0])
        c_pooled_2 = np.ones(data2.shape[0])

        args = [(data1, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_1),
                (data2, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_2)]

        if __name__ == '__main__':
            with multiprocessing.Pool(processes=2) as p:

        # LIST OF RETURNED DATAFRAMES
                results = p.starmap(cython_dpmm_algorithm1, args)



    if numdatasplit == 4 :
        data1, data2, data3, data4 = np.array_split(data.sample(frac=1), 4)
        c_pooled_1 = np.ones(data1.shape[0])
        c_pooled_2 = np.ones(data2.shape[0])
        c_pooled_3 = np.ones(data3.shape[0])
        c_pooled_4 = np.ones(data4.shape[0])

        args1 = [(data1, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_1),
                (data2, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_2)]

        args2 = [(data3, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_3),
                (data4, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled_4)]

        if __name__ == '__main__':
            with multiprocessing.Pool(processes=2) as p:

        # LIST OF RETURNED DATAFRAMES
                results1 = p.starmap(cython_dpmm_algorithm1, args1)

        time.sleep(len(data)/(100))

        if __name__ == '__main__':
            with multiprocessing.Pool(processes=2) as p:

        # LIST OF RETURNED DATAFRAMES
                results2 = p.starmap(cython_dpmm_algorithm1, args2)


        results = np.concatenate((results1,results2), axis = 1)


    for i in range(len(results)):
        if i != 0:
            results[i] = results[i]+(i*len(np.unique(results[i-1])))

    clusters = np.concatenate(results, axis = 0)

    results = cython_dpmm_algorithm1(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, clusters)

    if savetime :
        elapsed_time = time.time() - start_time

    if savetime :
        return results, elapsed_time

    if savetime == False :
        return results
