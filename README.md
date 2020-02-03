# PROJET ELTDM - README

Ce dossier contient les notebooks, scripts et jeux de données nécessaires à la réplication des expériences.

## Contenu des dossiers

Le contenu des dossiers est le suivant : 

Le dossier ```data``` contient les données nécessaires à l'exécution de l'algorithme. 

Le dossier ```figures```contient les figures générées au cours du processus. 

Le dossier ```Fonction_et_utilities``` contient une version compilée de l'implémentation cython de l'algorithme. 

Le dossier ```lib``` contient les fonctions nécéssaires à l'exécution de l'algorithme ainsi qu'à la génération des données. 

Le dossier ```notebooks```contient les cinq notebooks qui mettent en place notre approche. 

Le dossier ```Rapport```contient le rapport de notre projet sous format PDF. 

## Contenu des notebooks

Les cinq notebooks décrivent l'implémentation de notre approche. 

Le premier notebook, ```1_benchmark_initial``` définit et présente l'algorithme et l'implémente de manière naïve, sans optimisation.

Le deuxième notebook, ```2_banc_d_essai``` introduit les outils de mesure de la performance de l'algorithme et les applique une première fois sur une version réécrite mais non optimisée de l'algorithme.

Le troisième notebook, ```3_version_cythonisee``` met en avant l'amélioration des performances de l'algorithme lorsque des composantes de ce dernier ont été réécrites en C.

Le quatrième notebook, ```4_cython_multiprocessing``` montre les gains qui peuvent être réalisés en parallélisant certaines tâches de l'algorithme.

Le cinquième notebook ```5_comparaison_et_conclusion``` résume les gains successifs de performance réalisés en améliorant notre approche. 

## Remarque

Le code ```.pyx```a été compilé sur MacOS 10.13.6. Il peut être nécessaire de le re-compiler pour pouvoir l'utiliser sur d'autres systèmes d'exploitation. 

## Performance

La cythonisation et la parallélisation ont permis un gain non négligeable en terme de durée d'execution comme en témoigne les graphiques suivants. On remarque un gain d'un facteur 20 entre la version non optimisée et la version parallélisée de l'algorithme, que l'on s'intéresse à la performance en fonction du nombre d'itérations (voir ci-dessous) ou du nombre d'observations (voir ci-après).

![alt text](https://github.com/hugothimonier/parallelization_sort/blob/master/rendu_final/figures/iterations_comparison.png)

![alt text](https://github.com/hugothimonier/parallelization_sort/blob/master/rendu_final/figures/observations_comparison.png)

