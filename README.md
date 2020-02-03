# PROJET ELTDM - README

Ce dossier contient les notebooks, scripts et jeux de données nécessaires à la réplication des expériences. 


## Introduction

L'objectif de ce projet a été de paralléliser un algorithme de Gibbs permettant de clusteriser des observations en supposant que ces dernières suivent un modèle DP-GMM. Nous avons procédé comme suit : après avoir écrit un algorithme permettant de clusteriser les données, nous l'avons réécrit une première fois en fonctionnalisant ses composantes cruciales en termes de performance, puis nous avons réécrit ces fonctions en C puis parallilisé l'exécution de ces dernières. 

Les différents notebooks présentent en détail le fonctionnement de l'algorithme ainsi que les grandes étapes de notre démarche.  Au terme de cette dernière, nous sommes parvenus à améliorer les performances d'un facteur 10 (voir notebook 5 ou section "Performance" ci-après).

## Présentation de l'algorithme

La figure ci-dessous résume l'idée principale de l'algorithme : nous avons des observations réparties selon $K$ cluster où $K$ est inconnu (l'approche bayésienne non paramétrique diffère en cela des approches usuelles telles que les k-means en ce qu'elle ne nécessite pas de spécifier un nombre de clusters au préalable). L'implémentation se fait en utilisant un algorithme de Gibbs, présenté ci-dessous. 

![alt text](https://github.com/hugothimonier/parallelization_sort/blob/master/rendu_final/figures/iterations_comparison.png)


## Contenu des dossiers

Le contenu des dossiers est le suivant : 

Le dossier ```data``` contient les données nécessaires à l'exécution de l'algorithme. 

Le dossier ```figures```contient les figures générées au cours du processus. 

Le dossier ```Fonction_et_utilities``` contient une version compilée de l'implémentation cython de l'algorithme. 

Le dossier ```lib``` contient les fonctions nécéssaires à l'exécution de l'algorithme ainsi qu'à la génération des données. 

Le dossier ```notebooks```contient les cinq notebooks qui mettent en place notre approche. 

Le dossier ```rapport```contient le rapport de notre projet sous format PDF ainsi qu'un sous dossier ```markdown``` contenant les notebooks au format ```.md``` si jamais les notebooks venaient à ne pas s'exécuter.

## Contenu des notebooks

Les cinq notebooks décrivent l'implémentation de notre approche. 

Le premier notebook, ```1_benchmark_initial``` définit et présente l'algorithme et l'implémente de manière naïve, sans optimisation.

Le deuxième notebook, ```2_banc_d_essai``` introduit les outils de mesure de la performance de l'algorithme et les applique une première fois sur une version réécrite mais non optimisée de l'algorithme.

Le troisième notebook, ```3_version_cythonisee``` met en avant l'amélioration des performances de l'algorithme lorsque des composantes de ce dernier ont été réécrites en C.

Le quatrième notebook, ```4_cython_multiprocessing``` montre les gains qui peuvent être réalisés en parallélisant certaines tâches de l'algorithme.

Le cinquième notebook ```5_comparaison_et_conclusion``` résume les gains successifs de performance réalisés en améliorant notre approche. 

## Réplication

Tous les documents et scripts nécessaires à la réplication de ce projet sont disponibles dans ce répertoire. Dans le dossier ```lib```, le fichier ```dpmm_optimized.py```contient la première réécriture de l'algorithme tandis que le fichier ```utilities.py``` contient la version initiale ainsi que diverses fonctions nécessaires à la génération et la visualisation des données. Dans le dossier ```Fonction_et_utilities```se trouvent les fonctions réécrites en C (dans le fichier ```dpmm_cython_functions.pyx```qui a ensuite été compilé) ainsi que les scripts ```dpmm_optimized_cython.py```et ```dpmm_optimized_cython_parallel.py```qui implémentent ces fonctions. 

### Remarque

Le code ```.pyx```a été compilé sur MacOS 10.13.6. Il peut être nécessaire de le re-compiler pour pouvoir l'utiliser sur d'autres systèmes d'exploitation. 

## Performance

La cythonisation et la parallélisation ont permis un gain non négligeable en terme de durée d'execution comme en témoigne les graphiques suivants. On remarque un gain d'un facteur 10 entre la version non optimisée et la version parallélisée de l'algorithme, que l'on s'intéresse à la performance en fonction du nombre d'itérations (voir ci-dessous) ou du nombre d'observations (voir ci-après).

![alt text](https://github.com/hugothimonier/parallelization_sort/blob/master/rendu_final/figures/iterations_comparison.png)

![alt text](https://github.com/hugothimonier/parallelization_sort/blob/master/rendu_final/figures/observations_comparison.png)

