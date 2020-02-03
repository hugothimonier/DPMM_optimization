# coding: utf-8


# Library imports
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

# Contient les fonctions utilisées pour générer et afficher les données simulées
# Les données sont simulées à partir de paramètres saisis par l'utilisateur

def generate_cluster_parameters(mu_0, Sigma_0, sigma_0, nu, K, isUniform):
    """
    Génère un vecteur de moyenne et une matrice de variance covariance
    pour chaque cluster.
    La matrice de covariance est sdp, ses composantes sont tirées uniformément
    dans ]O,sigma_max]
    Convention : on entre sigma^2 (donc le terme de covariance est passé à la racine)
    Le vecteur mu est un vecteur dont ses composantes sont tirées uniformément dans
    [-mu_max, mu_max].

    Arguments:
    - `mu_0`: la moyenne a priori des moyennes
    - `Sigma_0`: la matrice de variance-covariance a priori des moyennes

    - `sigma_0`: le paramètre d'échelle a priori pour les variances
    - `nu`: le nombre de degrés de liberté pour la distribution des variances

    - `K` le nombre de clusters à créer
    - `isUniform`: un booléen indiquant si les poids des clusters sont uniformes ou non.

    """

    # Initialier la matrice des paramètres.
    parameters = {}

    # Définition d'un vecteur de poids
    if isUniform:
        weights = np.full( K, 1 / K)
    else:
        # Do and order K uniform draws
        cumulative_weights = np.random.uniform(0,1,size = K-1) # Sample
        cumulative_weights = np.append(cumulative_weights,1.0) # Add 1
        cumulative_weights = np.append(cumulative_weights,0.0) # Add 0
        cumulative_weights.sort() # Sort

        # Deduce the vector of weights
        weights = np.array([cumulative_weights[i+1]-cumulative_weights[i] for i in range(len(cumulative_weights)-1)])



    # Pour chaque cluster, création des paramètres associés
    for i in range(1,K + 1):
        parameters[i] = {}

        # Sampler la matrice de variance-covariance.
        S = sigma_0 * np.eye(2) # Paramètre d'échelle.
        cov = invwishart.rvs(nu, S) # Valeur.

        # Sampler le vecteur de moyenne.
        mu = np.random.multivariate_normal(mu_0, Sigma_0)

        # Assigner à la clef i sa moyenne et sa matrice de covariance.
        parameters[i] = [mu, cov]

    return parameters, weights


def generate_observations(N, parameters_values, weights):
    """
    Cette fonction génère des observations à partir d'un dictionnaire de valeurs.

    Arguments:
    - `N`: le nombre d'observations à générer
    - `parameters_values`: dictionnaire contennant les paramètres des Gaussiennes
    pour chacun des clusters.
    - `weights`: un vecteur de poids pour les
    """

    # Retrieve the number of clusters
    K = max(parameters_values.keys())


    # Generate N uniform draws over 1,...K to assign each observation to a cluster.
    clusters = np.random.randint(1, K + 1, N).tolist()

    # Initialisation du dictionnaire et de ses clefs
    observations = {}
    for cluster in range(1,K + 1):

        # Initialiser la clef correspondant au cluster k
        observations[cluster] = {}

        # Filtrer les observations et générer les tirages nécessaires
        nb_obs = clusters.count(cluster)
        mean, cov = parameters_values[cluster]
        try:
            x_coord, y_coord = np.random.multivariate_normal(mean, cov, nb_obs).T
        except RuntimeWarning:
            einsum('ii->i', cov)[:] += 1e-7
            x_coord, y_coord = np.random.multivariate_normal(mean, cov, nb_obs).T

        # Stocker les valeurs des observations assignées au cluster k
        observations[cluster] = (x_coord.tolist(), y_coord.tolist())

    return observations


def convert_to_df(observations):
    """
    Convertit un dictionnaire contenant des coordonnées d'observations
    en une DataFrame où chaque ligne correspond à une observation,
    la première colone à la coordonnée x et la seconde à la coordonnée y.

    Argument:
    - ‘observations‘: un dictionnaire dans lequel chaque clef contient
    deux lists, une pour chaque coordonnée x et y.
    """

    # Faire deux listes de données.
    x_coords = []
    y_coords = []
    cluster = []
    for key in observations.keys():
        # On instancie un vecteur qui répète la valeur du cluster

        # On ajoute à la liste les coordonnées associées avec le cluster courant
        x_coords += observations[key][0]
        y_coords += observations[key][1]
        cluster += [key for _ in range(len(observations[key][0]))]

    # On créé la df à partir des deux listes de coordonnées
    obs_df = pd.DataFrame(list(zip(x_coords,y_coords, cluster)), columns = ['x_coordinate', 'y_coordinate', 'cluster'])

    # On randomise les lignes du tableau
    obs_df = obs_df.sample(frac=1).reset_index(drop=True)

    return obs_df


def plot_clustered_observations(observations):
    """
    Affiche les observations par cluster. Chaque cluster est identifié par une
    couleur.

    Arguments:
    - `observations`: un dictionnaire d'observations.
    """

    for key in observations.keys():

        # On tire la couleur du cluster.
        colors = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))

        # On déballe les coordonnées x et y.
        x_axis = observations[key][0]
        y_axis = observations[key][1]

        # On crée le plot.
        plt.scatter(x_axis, y_axis, c=colors)

    # Une fois toutes les clefs parcourues, on affiche.
    plt.rcParams["figure.figsize"] = (30,18)
    plt.show()

    return None


def dpmm_algorithm0(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_init, savetime = False):
    """
    Cette fonction simule l'assignation aux clusters selon un DP-GMM pour un
    nombre potentiellement infini de clusters.

    En pratique, on implémente un DP-MM sous sa représentation CRP.

    Il retourne une matrice (n_obs, n_iter), où chaque colonne correspond
    au cluster assigné à l'observation i ∈ 0,...,n_obs. La dernière colonne
    correspond donc à la dernière assignation.
    Et également le temps mis pour executer le code

    La première colonne correspond à la première assignation aribitraire,


    Les moyennes et covariances estimées des clusters aussi ?

    Arguments:
    - `data`: une DataFrame contenant nxk les observations (n : nombre d'observations
              et k dimension des observations).
    - `alpha`: le paramètre de concentration. Plus alpha est élevé, plus facilement
             de nouveaux clusters seront crées.
    - `mu_0` : la moyenne a priori de la distribution des clusters.
    - `Sigma_0`: la variance a priori de la distribution des clusters.
    - `sigma_x`: la variance des observations, supposée être sigma_x * Identité
    - `n_iter`: le nombre maximal d'itérations.
    - `c_init`: le vecteur correspondant aux assignations initiales.

    """
    assert savetime in [True, False]

    if savetime :
        start_time = time.time()


    # Convert in the form of an array
    data = data.values
    # Retrieve the dimensions of the data
    n_obs, data_dim = data.shape

    # Priors

    # Precision parameters
    tau_0 = inv(Sigma_0)
    Sigma_x = sigma_x * np.eye(data_dim)
    tau_x = 1/sigma_x * np.eye(data_dim)

    # Initial cluster assignment
    z = c_init.astype(int)

    # Counts per cluster : dictionnary with key = label, value = count
    unique, counts = np.unique(z, return_counts=True)
    n_k = dict(zip(unique, counts))

    # initial number of clusters
    n_clust = len(n_k.keys())

    # Initialization of the matrix of cluster membership.
    res =  np.empty((n_obs, n_iter))

    # Initialize a progress bar (function quite long to run)
    pbar = ProgressBar()

    # Beginning of the algorithm
    for iter in pbar(range(n_iter)):
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

            # Log probabilities for the clusters
            logp = np.empty(n_clust + 1)

            # Loop over the non-empty clusters
            for c_i in n_k.keys():


                tau_p = tau_0 + n_k[c_i] * tau_x # Cluster precision
                Sigma_p = inv(tau_p) # Cluster variance

                # Find the points belonging to that cluster
                loc_z = np.where(z == c_i)
                # Sum the points of this cluster.
                sum_data = np.sum(data[loc_z], axis = 0)

                # Compute the predictive distribution in the already occupied
                # tables.
                mean_p = np.dot(Sigma_p , (np.dot(tau_x,sum_data) + np.dot(tau_0,mu_0)))

                # Evaluate with these parameters (in log)
                # dimension mismatch are likely
                logp[c_i-1] = math.log(n_k[c_i]) + multivariate_normal.logpdf(data[n,:], mean= mean_p, cov = Sigma_p + Sigma_x)

            # Compute the predictive probability of belonging to a new cluster.
            logp[n_clust] = math.log(alpha) + multivariate_normal.logpdf(data[n,:], mean = mu_0, cov = Sigma_0 + Sigma_x)

            # log-sum-exp trick and convert into probabilities
            max_logp = max(logp)
            logp = logp - max_logp
            loc_probs = np.exp(logp)
            sum_loc_probs = np.sum(loc_probs)
            loc_probs = np.array([locp / sum_loc_probs for locp in loc_probs])
            #print('loc probabilities for observation %s at iteration %s' %(n,iter))
            #print(loc_probs)

            # Draw a sample of which cluster the new observation should belong to
            # dimensions
            newz = int(np.random.choice(n_clust+1, 1, p = loc_probs))
            newz += 1   # increase to translate the outcomte into a cluster label

            # Spawn a new cluster if necessary
            if newz == n_clust + 1:
                n_k[newz] = 0 # initialize a new key in the dictionnary.
                n_clust += 1 # increase the number of clusters.

            z[n] = newz # assign the new cluster value to the obs.
            n_k[newz] += 1 # increase the number of observation in the cluster.

        # Write the output of the current iteration in the output matrix
        res[:,iter] = z

    if savetime :
        elapsed_time = time.time() - start_time

    if savetime :
        return res, elapsed_time

    if savetime == False :
        return res



def dpmm_algorithm_multiprocessing(data, numdatasplit, alpha, mu_0, Sigma_0, sigma_x, n_iter, savetime = False, __name__ = '__main__'):

    assert numdatasplit in [2,4], 'Le nombre de data split doit être 2 ou 4'
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
                results = p.starmap(dpmm_algorithm, args)



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
                results1 = p.starmap(dpmm_algorithm, args1)

        time.sleep(len(data)/(100))

        if __name__ == '__main__':
            with multiprocessing.Pool(processes=2) as p:

        # LIST OF RETURNED DATAFRAMES
                results2 = p.starmap(dpmm_algorithm, args2)


        results = np.concatenate((results1,results2), axis = 1)


    for i in range(len(results)):
        if i != 0:
            results[i] = results[i]+(i*len(np.unique(results[i-1])))

    clusters = np.concatenate(results, axis = 0)[:,-1]

    results = dpmm_algorithm(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, clusters)

    if savetime :
        elapsed_time = time.time() - start_time

    if savetime :
        return results, elapsed_time

    if savetime == False :
        return results


def plot_clusters_from_dataframe(results):
    """
    Plots for the two-dimensional case the matrix of observations
    and their associated cluster (i.e. one color per cluster)

    Different from the other function, this one generates the dictionnary from
    a DataFrame and then does the plotting.

    The dictionnary is organized as follows :
    keys : cluster numbers
    values : a tuple of (x_coord_list, y_coord_list) where
    x_coord_list, y_coord_list correspond to the coordinates of the
    observations falling into the cluster n°key.

    Arguments:
    - `results`: a DataFrame of results, the last column should correspond to the
    cluster number
    """

    # Get the list of clusters. They will be the keys of the dictionnary
    observed_clusters = results.iloc[ : , 2 ].unique()
    cluster_count = len(observed_clusters)


    # Initialize the dictionnary
    observations  = {}

    for cluster in observed_clusters:
        # Retrieve observations corresponding to the current cluster
        matching_observations = results.loc[results.iloc[:, 2] == cluster]

        # Keep only the two first columns (x and y coordinates)
        matching_coords = matching_observations.iloc[:,0:2].values

        # Transpose and convert to a list
        matching_coords = np.transpose(matching_coords).tolist()

        # Now the first row are the x's and the second the y's.
        # We can directly plug them into the dictionnary.
        observations[cluster] = (matching_coords[0], matching_coords[1])


    # Plot from the created dictionnary
    plot_clustered_observations(observations)

    return cluster_count

def reconstruct_clusters(coordinates, clusters):
    """
    Constructs a DataFrame with the third column corresponding
    to the cluster that have been assigned to the observation.

    Only works in the 2d-case. Only the last assignation is considered.

    Returns a dataframe with three columns : x_coord, y_coord and cluster

    Arguments:
    - `coordinates` : a n_obs*2 DataFrame of coordinates (x and y)
    - `clusters` : a n_obs*n_iter array of cluster values.
    """

    # Indices in the array of observations and the clusters match (by construction)
    # so we can just concatenate

    # Convert the dataframe into two 1D arrays
    data_x = coordinates.values[:,0]
    data_y = coordinates.values[:,1]


    # Retrieve the last column of the cluster assignations.
    last_locations = clusters[:,-1]

    # Create a n_obs*3 array
    reconstructed_clusters = np.column_stack((data_x,data_y,last_locations))

    # Column names for the new dataframe
    colnames = ["x_coordinate", "y_coordinate", "assignated_clusters"]

    estimated_clusters = pd.DataFrame(reconstructed_clusters)
    estimated_clusters.columns = colnames


    return estimated_clusters


def cluster_count_evolution(clusters):
    """
    Plots the evolution of the number of clusters as a function of the number
    of iterations.

    Argument:
    - `clusters` : a n_obs * n_iter array
    """

    # Get the number of steps
    n_steps = clusters.shape[1]

    # Initialize the vector of cluster count
    clusters_count = np.empty(n_steps)

    # For each column, get the number of clusters.
    for iter in range(n_steps):

        # Current column
        current_vals = clusters[:,iter]
        # Count the number of unique values
        clusters_count[iter] = len(np.unique(current_vals, return_counts=False))

    # Do the plot and return it
    plt.plot(range(n_steps),clusters_count)
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of clusters")
    plt.show()

    return None


def return_clustered_observations(observations):
    """
    Autre methode, renvoie le plot que l'on peut modifier ensuite.

    Affiche les observations par cluster. Chaque cluster est identifié par une
    couleur.

    Arguments:
    - `observations`: un dictionnaire d'observations.
    """

    # Initialisation de la liste qui contiendra les couleurs
    clr = []
    for key in observations.keys():

        # On tire la couleur du cluster.
        colors = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
        clr.append(colors)

        # On déballe les coordonnées x et y.
        x_axis = observations[key][0]
        y_axis = observations[key][1]

        # On crée le plot.
        plt.scatter(x_axis, y_axis, c=colors)

    return plt, clr


def return_clusters_from_dataframe(results):
    """
    Méthode alternative, retourne le plot.

    Plots for the two-dimensional case the matrix of observations
    and their associated cluster (i.e. one color per cluster)

    Different from the other function, this one generates the dictionnary from
    a DataFrame and then does the plotting.

    The dictionnary is organized as follows :
    keys : cluster numbers
    values : a tuple of (x_coord_list, y_coord_list) where
    x_coord_list, y_coord_list correspond to the coordinates of the
    observations falling into the cluster n°key.

    Arguments:
    - `results`: a DataFrame of results, the last column should correspond to the
    cluster number
    """

    # Get the list of clusters. They will be the keys of the dictionnary
    observed_clusters = results.iloc[ : , 2 ].unique()
    cluster_count = len(observed_clusters)


    # Initialize the dictionnary
    observations  = {}

    for cluster in observed_clusters:
        # Retrieve observations corresponding to the current cluster
        matching_observations = results.loc[results.iloc[:, 2] == cluster]

        # Keep only the two first columns (x and y coordinates)
        matching_coords = matching_observations.iloc[:,0:2].values

        # Transpose and convert to a list
        matching_coords = np.transpose(matching_coords).tolist()

        # Now the first row are the x's and the second the y's.
        # We can directly plug them into the dictionnary.
        observations[cluster] = (matching_coords[0], matching_coords[1])


    # Plot from the created dictionnary
    plt, colors = return_clustered_observations(observations)

    return cluster_count, plt, colors


def mean_time(function, n_iter2, args):

    """

    Calcule le temps moyen d'execution d'une fonction

    Arguments :
    function : fonction à évaluer
    n_iter2 : nombre d'itérations sur lesquelles est calculée la moyenne
    args : les paramètres de la fonction évaluée, ce doit être soit la forme d'un tuple
            par exemple pour la fonction dpmm : (data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled)

    """

    time = []

    for i in range(n_iter2) :

        output, time_ = function(*args, True)
        time.append(time_)

    del output, time_

    mean = np.mean(time)

    return mean
