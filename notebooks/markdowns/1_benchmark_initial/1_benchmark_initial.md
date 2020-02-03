
# Elements logiciels  pour le traitement des données massives

<b> Projet de fin de semestre - année universitaire 2019-2020 

Hugo Thimonier, Gabriel Kasmi - 3A DS-SA </b>

# 1. Présentation de l'algorithme et benchmark

Dans ce premier notebook, nous présentons le problème, l'algorithme que nous utilisons et procédons à un premier benchemark sur sa performance. Dans les notebooks suivants, nous procéderons à une optimisation plus poussée de l'algorithme.


```python
from jyquickhelper import add_notebook_menu
add_notebook_menu()
```




<div id="my_id_menu_nb">run previous cell, wait for 2 seconds</div>
<script>
function repeat_indent_string(n){
    var a = "" ;
    for ( ; n > 0 ; --n)
        a += "    ";
    return a;
}
// look up into all sections and builds an automated menu //
var update_menu_string = function(begin, lfirst, llast, sformat, send, keep_item, begin_format, end_format) {
    var anchors = document.getElementsByClassName("section");
    if (anchors.length == 0) {
        anchors = document.getElementsByClassName("text_cell_render rendered_html");
    }
    var i,t;
    var text_menu = begin;
    var text_memo = "<pre>\nlength:" + anchors.length + "\n";
    var ind = "";
    var memo_level = 1;
    var href;
    var tags = [];
    var main_item = 0;
    var format_open = 0;
    for (i = 0; i <= llast; i++)
        tags.push("h" + i);

    for (i = 0; i < anchors.length; i++) {
        text_memo += "**" + anchors[i].id + "--\n";

        var child = null;
        for(t = 0; t < tags.length; t++) {
            var r = anchors[i].getElementsByTagName(tags[t]);
            if (r.length > 0) {
child = r[0];
break;
            }
        }
        if (child == null) {
            text_memo += "null\n";
            continue;
        }
        if (anchors[i].hasAttribute("id")) {
            // when converted in RST
            href = anchors[i].id;
            text_memo += "#1-" + href;
            // passer à child suivant (le chercher)
        }
        else if (child.hasAttribute("id")) {
            // in a notebook
            href = child.id;
            text_memo += "#2-" + href;
        }
        else {
            text_memo += "#3-" + "*" + "\n";
            continue;
        }
        var title = child.textContent;
        var level = parseInt(child.tagName.substring(1,2));

        text_memo += "--" + level + "?" + lfirst + "--" + title + "\n";

        if ((level < lfirst) || (level > llast)) {
            continue ;
        }
        if (title.endsWith('¶')) {
            title = title.substring(0,title.length-1).replace("<", "&lt;")
         .replace(">", "&gt;").replace("&", "&amp;");
        }
        if (title.length == 0) {
            continue;
        }

        while (level < memo_level) {
            text_menu += end_format + "</ul>\n";
            format_open -= 1;
            memo_level -= 1;
        }
        if (level == lfirst) {
            main_item += 1;
        }
        if (keep_item != -1 && main_item != keep_item + 1) {
            // alert(main_item + " - " + level + " - " + keep_item);
            continue;
        }
        while (level > memo_level) {
            text_menu += "<ul>\n";
            memo_level += 1;
        }
        text_menu += repeat_indent_string(level-2);
        text_menu += begin_format + sformat.replace("__HREF__", href).replace("__TITLE__", title);
        format_open += 1;
    }
    while (1 < memo_level) {
        text_menu += end_format + "</ul>\n";
        memo_level -= 1;
        format_open -= 1;
    }
    text_menu += send;
    //text_menu += "\n" + text_memo;

    while (format_open > 0) {
        text_menu += end_format;
        format_open -= 1;
    }
    return text_menu;
};
var update_menu = function() {
    var sbegin = "";
    var sformat = '<a href="#__HREF__">__TITLE__</a>';
    var send = "";
    var begin_format = '<li>';
    var end_format = '</li>';
    var keep_item = -1;
    var text_menu = update_menu_string(sbegin, 2, 4, sformat, send, keep_item,
       begin_format, end_format);
    var menu = document.getElementById("my_id_menu_nb");
    menu.innerHTML=text_menu;
};
window.setTimeout(update_menu,2000);
            </script>




```python
# Importation des libraires 
import random
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Importation du fichier contenant les fonctions nécessaires aux simulations
%run '../lib/utilities.py'
```

## Définition formelle du modèle

On considère le modèle hiérarchique suivant:
<ul> 
<li> $\pi = (\pi_1, \pi_2, ...) \sim GEM (\alpha)$, $\alpha >0$. $\pi_i$ correspond à la probabilité de tomber dans le cluster $i$, pour $i>0$. La notation "GEM" fait référence à la distribution Griffiths–Engen–McCloskey. 
</li>
<li> $\mu_k \overset{i.i.d.}{\sim}\mathcal{N}(\mu_0, \Sigma_0)$  </li>
<li> $\Sigma_k \overset{i.i.d.}{\sim}Inv-Wishart(S, \nu)$ où $S = \sigma_0 \times I_2 $ </li>
<li> $\forall i\in \{1,\dots, n\},\, z_i \overset{i.i.d.}{\sim}Categorical(\pi)$, de telle sorte que $Z\in\{1,2,\dots\}$ avec probabilités respectives $\pi_1,\,\pi_2, $ etc.</li>
<li> $\forall i\in \{1,\dots, n\},\, x_i \overset{indep}{\sim}\mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$</li>
</ul>
Seuls les $x_i$ sont observés. 

Notons que l'on peut réécrire le modèle en terme de processus de Dirichelet en notant que nos observations suivent un $DP(\alpha,P)$ où $P = \mathcal{N}(\mu_0,\Sigma_0)$ est la mesure de base du processus.  

Les hyperparamètres sont $\mu_0,\Sigma_0,\sigma_0,\nu$ et $\alpha$. 



Dans le modèle d'inférence, nous spécifierons en plus un paramètre $\sigma_x$ correspondant au bruit contenu dans les observations. 

## Simulation des données

### Modèle de simulation

Dans le modèle de simulation, le nombre de clusters $K$ n'est pas aléatoire mais choisi par l'utilisateur. Les poids des différents clusters sont déterminés aléatoirement ou alors calés uniforméments. 

Dans le premier cas, on réalise $K$ tirage d'une $\mathcal{U}[0,1]$ que l'on ordonne ensuite pour avoir les poids des clusters tels que $\sum_{k=1}^K \pi_k = 1$ . 

Dans le second, les poids $\pi_i = \pi = 1/K$

L'utilisateur spéficie également un nombre d'observations $n_{obs}$. Les hyperparamètres, décrits dans la section précédente sont initialisés par défaut avec les valeurs suivantes:

<ul>
<li> $\mu_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$ </li>
<li> $\Sigma_0 = \begin{pmatrix}32 & 0 \\ 0 & 32\end{pmatrix}$</li>
<li> $\nu = 9$</li>
<li> $\sigma_0$ = 1 de sorte que S = $I_2$</li>

</ul>




```python
# Paramètres entrés par l'utilisateur:

# Définition de la graine (pour la reproducibilité)
random.seed(42)

# Taille de l'échantillon.
n_obs = 250

# Distribution a priori des centroïdes des clusters :
mu_0 = np.array([0, 0]) # Espérance de la loi normale pour mu.
Sigma_0 = 32 * np.eye(2) # Variance (échelle) de la loi normale pour mu.

# Distribution a priori des variances des clusters : 
nu = 9.0 # Nombre de degrés de liberté de la loi inv-Wishart.
sigma_0 =  1. # "Variance" (échelle) pour la loi inv-Wishart.

# Paramètres spéficiques à la simulation.
K = 5 # Nombre de clusters.
isUniform = False # Indicatrice pour savoir si l'assignation aux clusters est uniforme ou non. 
                  # Dans le cas où l'assignation n'est pas uniforme, les poids des clusters sont 
                  # tirés aléatoirement. 
```


```python
# Générer les paramètres des lois des clusters (mu_k et sigma_k) selon des Gaussiennes et une inv-Wishart.
parameters, weights = generate_cluster_parameters(mu_0, Sigma_0, sigma_0, nu, K, isUniform)
# Afficher les poids pour chaque cluster
for par, w in zip(parameters.keys(), weights):
    print("Le poids associé au cluster %s est %s" % (par, w))
print("Les poids somment à %s" % (sum(weights)))
```

    Le poids associé au cluster 1 est 0.260103302412912
    Le poids associé au cluster 2 est 0.17496863446907596
    Le poids associé au cluster 3 est 0.20716592524307376
    Le poids associé au cluster 4 est 0.3338170354414227
    Le poids associé au cluster 5 est 0.023945102433515553
    Les poids somment à 1.0



```python
# Générer des observations et les ordonner par clusters
observations = generate_observations(n_obs, parameters, weights)
```

### Visualisation des données simulées


```python
# Afficher les observations simulées (exécuter la cellule deux fois pour avoir un meilleur affichage)
plot_clustered_observations(observations)
```


![png](output_16_0.png)



```python
# Conversion au format DataFrame et export en .csv des données simulées.
observations_dataframe = convert_to_df(observations).to_csv('../data/simulated_observations.csv', index = False)
```

On génère ensuite des jeux de 500 et 1000 observations. Ces jeux de données plus grands nous permettront de voir comment l'algorithme se comporte lorsque le nombre d'observations augmente. Le jeu de données de base contient 250 observations, nous en générons deux supplémentaires de 500 et 1000 observations respectivement.


```python
# 500 observations
observations = generate_observations(500, parameters, weights)
observations_dataframe = convert_to_df(observations).to_csv('../data/simulated_observations_500.csv', index = False)

# 1000 observations
observations = generate_observations(1000, parameters, weights)
observations_dataframe = convert_to_df(observations).to_csv('../data/simulated_observations_1000.csv', index = False)
```

## Présentation de l'algorithme et premier benchmark

Une manière courante de représenter le DP-GMM est d'utiliser la métaphore dire du restaurant chinois (CRP). Dans cette métaphore, les clusters sont des tables et pour chaque observation, la question est de savoir si elle doit rejoindre une table déjà existante ou bien être assignée à une nouvelle table. 

En utilisant la définition "chinese restaurant process" du DP-GMM, nous implémentons un algorithme de Gibbs pour assigner les observations à des clusters. Les tables sont identifiées par un couple $\phi_k:=(\mu_k,\tau_k)$ la moyenne et la précision du cluster. La probabilité de s'installer à une table est proportionnelle au nombre d'observations déjà assignées à cette table. 

Nous allons dans un premier temps calculer la probabilité <i> a posteriori </i> d'être assigné à un cluster $k$ puis nous en déduirons les probabilités a posteriori pour une observation d'appartenir à ce cluster. 

<b> Probabilité <i> a posteriori </i> d'être assigné à un cluster </b>

La probabilité d'être assigné à un cluster (on néglige pour l'instant la dimension spaciale donnée par la deuxième partie du modèle hiérarchique) est la suivante:
$$
\begin{aligned}
p(c_1,\dots,c_k\mid \alpha ) &= \displaystyle{\int p(c_1,\dots,c_k\mid \pi_1,\mid,\pi_k ) \times p(\pi_1,\dots,\pi_k\mid\alpha)\mathrm{d}\pi_1\dots\mathrm{d}\pi_k}\\
&= \displaystyle{\frac{\Gamma(\alpha)}{\Gamma\left(\alpha/K\right)^k}\int \prod_{k=1}^K\pi^{n_k+\alpha/K-1}\mathrm{d}\pi_k
}\\
&= \displaystyle{\frac{\Gamma(\alpha)}{\Gamma(n+\alpha)}\prod_{k=1}^K\frac{\Gamma(n_k+\alpha/K)}{\Gamma(\alpha/K)}
}\\
\end{aligned}
$$
D'où l'on peut finalement déduire que $\displaystyle{p(c_i = k\mid c_{-i},\alpha) = \frac{n_{-i,k}+\alpha/K}{n-1+\alpha}
}$ où $x_{-i}$ désigne le vecteur $x$ privé de la $i$ème coordonnée. Enfin, en prenant pour $K\to\infty$, on obtient que la probabilité d'assignation à un cluster est
$$
\begin{array}{ll}
\textrm{pour les clusters où }n_{-i,k} > 0 \;:\;& p(c_i\mid c_{-i},\alpha) = \displaystyle{\frac{n_{-i,k}}{n-1+\alpha}} \\
\textrm{pour tous les autres clusters } :\;& p(c_j\neq c_k \forall j\neq i\mid c_{-i},\alpha) = \displaystyle{\frac{\alpha}{n-1+\alpha}}
\end{array}
$$
On vérifie bien que la probabilité d'assignation est proportionnelle au facteur $\alpha$.


<b> Probabilité <i> a posteriori </i> d'appartenir à un cluster </b>

L'idée dans le CRP est de traiter les observations séquentiellement. On commence par suppose que chaque observation $x\sim \mathcal{N}(\mu,\sigma_x^2)$. Compte tenu du fait que $\mu\sim\mathcal{N}(\mu_0,\sigma_0^2)$, on a finalement que $\tilde{x}\sim\mathcal{N}(\mu_0,\sigma_0^2 + \sigma_x^2)$. Cela nous donne la densité <i> a posteriori </i> d'une observation ne se rattachant à aucun cluster. 

Pour une observation qui se rattache à un cluster, nous avons $\displaystyle{\tilde{x}\sim\mathcal{N}\left(\underbrace{\frac{\bar x_k n_k \tau_k + \mu_0\tau_0 }{n_k\tau_k+\tau_0}}_{\mu_p},\underbrace{\frac{1}{n_k\tau_k+\tau_0}}_{\sigma^2_p}+\sigma_x^2
\right)
}$ où $\tau$ désigne la précision (donc $\tau = 1/\sigma$). Nous utilisons le fait que le modèle est conjugué. Les détails des calculs sont présentés dans le rapport. 

On peut donc résumer la distribution <i> a posteriori </i> des $c_i$:
$$
\begin{array}{ll}
\textrm{pour les clusters où }n_{-i,k} > 0 \;:\;& p(c_i\mid c_{-i},\mu_p,\tau_p,\alpha)\propto p(c_i\mid c_{-i}\alpha)\times p(\tilde{x}\mid \mu_p, \tau_p, c_{-i}) \propto \displaystyle{\frac{n_{-i,k}}{n-1+\alpha} \times \mathcal{N}(\tilde{x_i};\mu_p,\sigma_p^2+\sigma_x^2)} \\
\textrm{pour tous les autres clusters } :\;& p(c_j\neq c_k \forall j\neq i\mid c_{-i}, \tau_0, \mu_0, \alpha) \propto \displaystyle{\frac{\alpha}{n-1+\alpha}\times \mathcal{N}(\tilde{x_j};\mu_0,\sigma_0^2+\sigma_x^2)}
\end{array}
$$
Où la notation $\mathcal{N}(x;\mu,\sigma^2)$ désigne l'évaluation de la densité d'une $\mathcal{N}(\mu,\sigma^2)$ en $x$.

Nous pouvons finalement en déduire l'algorithme suivant: 

<ul>
<li> Input : $\alpha$, $\mu_0$, $\sigma_0^2$, $\sigma_x^2$ </li>
<li> Pour $i \le n_{iter}$ </li>
<ul> 
<li> Pour $j \le n_{obs}$ </li>
<ul>
<li> Enlever $x_i$ de son cluster car nous allons le rééchantillonner </li>
<li> Echantillonner $c_i \mid c_{-i}, x$ de la manière suivante: </li>
<li> Pour $k \le N_{cluster} +1 $ </li>
<ul>
<li> Calculer la probabilité que $c_i = k$ d'après la formule suivante :</li> 
<li> $p(c_i= k\mid c_{-i},\mu_p,\tau_p,\alpha)\propto \frac{n_{-i,k}}{n-1+\alpha} \times \mathcal{N}(\tilde{x_i};\mu_p,\sigma_p^2+\sigma_x^2)$</li> 

<li> Calculer la probabilité que $c_i = N_{cluster} + 1$ d'après la formule suivante :</li> 
<li> $p(c_j\neq c_k \forall j\neq i\mid c_{-i}, \tau_0, \mu_0, \alpha) \propto \frac{\alpha}{n-1+\alpha}\times \mathcal{N}(\tilde{x_j};\mu_0,\sigma_0^2+\sigma_x^2)$ </li>
</ul>
<li> $c^{i} = c_j$, le cluster échantillonné devient le "nouveau" cluster de l'observation. </li>
</ul>
</ul>
</ul>
<li> Output : renvoyer $c$. </li>



### Une première exécution

Dans cette section, nous exécutons l'algorithme en conditions "réelles" et observons le temps d'exécution, les threads et le profil des fonctions appelées. Dans les notebooks suivants, nous optimiserons cet algorithme et analyseront les résultats de manière plus fine. 


```python
# Paramètres complémentaires pour l'inférence.
alpha = 0.03 # Paramètre d'échelle pour le processus de Dirichelet.
sigma_x = 0.5 # Bruit sur les observations.


# Nombre d'itérations pour l'algorithme de Gibbs :
n_iter = 1000
```


```python
# Importation du CSV des observations simulées
# Si on veut charger le fichier "sauvegardé" avec 5 clusters, commenter la ligne 3 et décommenter la ligne 4.
raw_input = pd.read_csv('../data/simulated_observations.csv')
# raw_input = pd.read_csv('simulated_observations_5_clusters.csv')
data = raw_input[["x_coordinate",'y_coordinate']]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_coordinate</th>
      <th>y_coordinate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.378057</td>
      <td>4.933422</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-8.385058</td>
      <td>4.309316</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.997994</td>
      <td>6.233421</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.723696</td>
      <td>1.696573</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.614551</td>
      <td>5.210928</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%time
# Les paramètres pour l'inférence ont été définis en début de section. 
# Pour les paramètres des lois a priori, on reprend les mêmes que ceux qui ont généré les données. 

# Pour l'assignation des clusters, on a le choix. On commence avec une assignation dans un même cluster. 
c_pooled = np.ones(data.shape[0])
# On exécute l'algorithme
assignations_pooled = dpmm_algorithm0(data, alpha, mu_0, Sigma_0, sigma_x, n_iter, c_pooled)
```

    100% |########################################################################|

    CPU times: user 10min 40s, sys: 1min 2s, total: 11min 43s
    Wall time: 6min 33s


    


### Analyse des threads

On réduit le nombre d'itérations de l'algorithme à 1 et on analyse plus précisément les fonctions appelées, les temps d'éxecution. 


```python
import cProfile
cProfile.run('dpmm_algorithm0(data, alpha, mu_0, Sigma_0, sigma_x, 1, c_pooled)')

```

      0% |                                                                        |

    500


    100% |########################################################################|


             357258 function calls (357253 primitive calls) in 0.847 seconds
    
       Ordered by: standard name
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         2489    0.003    0.000    0.036    0.000 <__array_function__ internals>:2(amax)
         2489    0.002    0.000    0.024    0.000 <__array_function__ internals>:2(amin)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(concatenate)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(diff)
         8456    0.007    0.000    0.029    0.000 <__array_function__ internals>:2(dot)
         1990    0.002    0.000    0.098    0.000 <__array_function__ internals>:2(inv)
         2489    0.003    0.000    0.009    0.000 <__array_function__ internals>:2(iscomplexobj)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(nonzero)
          500    0.001    0.000    0.009    0.000 <__array_function__ internals>:2(prod)
         7467    0.008    0.000    0.101    0.000 <__array_function__ internals>:2(sum)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(unique)
         1989    0.002    0.000    0.009    0.000 <__array_function__ internals>:2(where)
            1    0.000    0.000    0.847    0.847 <string>:1(<module>)
        14435    0.005    0.000    0.020    0.000 _asarray.py:16(asarray)
            3    0.000    0.000    0.000    0.000 _asarray.py:88(asanyarray)
            1    0.000    0.000    0.000    0.000 _collections_abc.py:657(get)
         2489    0.001    0.000    0.015    0.000 _methods.py:47(_all)
         2489    0.014    0.000    0.015    0.000 _multivariate.py:104(<listcomp>)
         2489    0.062    0.000    0.401    0.000 _multivariate.py:147(__init__)
         2489    0.010    0.000    0.023    0.000 _multivariate.py:359(_process_parameters)
         2489    0.004    0.000    0.006    0.000 _multivariate.py:40(_squeeze_output)
         2489    0.005    0.000    0.007    0.000 _multivariate.py:419(_process_quantiles)
         2489    0.041    0.000    0.079    0.000 _multivariate.py:437(_logpdf)
         2489    0.017    0.000    0.532    0.000 _multivariate.py:464(logpdf)
         2489    0.016    0.000    0.063    0.000 _multivariate.py:52(_eigvalsh_to_eps)
         2489    0.006    0.000    0.032    0.000 _multivariate.py:87(_pinv_1d)
         2489    0.017    0.000    0.054    0.000 _util.py:192(_asarray_validated)
           10    0.000    0.000    0.000    0.000 _weakrefset.py:70(__contains__)
            6    0.000    0.000    0.000    0.000 abc.py:180(__instancecheck__)
            1    0.000    0.000    0.000    0.000 arraysetops.py:138(_unpack_tuple)
            1    0.000    0.000    0.000    0.000 arraysetops.py:146(_unique_dispatcher)
            1    0.000    0.000    0.000    0.000 arraysetops.py:151(unique)
            1    0.000    0.000    0.000    0.000 arraysetops.py:297(_unique1d)
         2489    0.002    0.000    0.003    0.000 base.py:1202(isspmatrix)
            1    0.000    0.000    0.000    0.000 base.py:641(__len__)
         2489    0.013    0.000    0.030    0.000 blas.py:218(find_best_blas_type)
         2489    0.002    0.000    0.002    0.000 blas.py:259(<listcomp>)
         2489    0.020    0.000    0.058    0.000 blas.py:279(_get_funcs)
            1    0.000    0.000    0.000    0.000 common.py:89(is_object_dtype)
         2489    0.002    0.000    0.002    0.000 core.py:6251(isMaskedArray)
         2489    0.060    0.000    0.186    0.000 decomp.py:240(eigh)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:1755(_nonzero_dispatcher)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:1759(nonzero)
         7467    0.002    0.000    0.002    0.000 fromnumeric.py:2040(_sum_dispatcher)
         7467    0.013    0.000    0.084    0.000 fromnumeric.py:2045(sum)
         2489    0.001    0.000    0.001    0.000 fromnumeric.py:2499(_amax_dispatcher)
         2489    0.004    0.000    0.029    0.000 fromnumeric.py:2504(amax)
         2489    0.001    0.000    0.001    0.000 fromnumeric.py:2624(_amin_dispatcher)
         2489    0.003    0.000    0.019    0.000 fromnumeric.py:2629(amin)
          500    0.000    0.000    0.000    0.000 fromnumeric.py:2787(_prod_dispatcher)
          500    0.001    0.000    0.008    0.000 fromnumeric.py:2792(prod)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:55(_wrapfunc)
        12945    0.029    0.000    0.115    0.000 fromnumeric.py:73(_wrapreduction)
        12945    0.010    0.000    0.010    0.000 fromnumeric.py:74(<dictcomp>)
            1    0.000    0.000    0.000    0.000 function_base.py:1143(_diff_dispatcher)
            1    0.000    0.000    0.000    0.000 function_base.py:1147(diff)
         2489    0.013    0.000    0.032    0.000 function_base.py:432(asarray_chkfinite)
            1    0.000    0.000    0.000    0.000 generic.py:4378(__setattr__)
            1    0.000    0.000    0.000    0.000 generic.py:4423(_protect_consolidate)
            1    0.000    0.000    0.000    0.000 generic.py:4433(_consolidate_inplace)
            1    0.000    0.000    0.000    0.000 generic.py:4436(f)
            1    0.000    0.000    0.000    0.000 generic.py:4563(values)
         3489    0.006    0.000    0.007    0.000 getlimits.py:365(__new__)
            1    0.000    0.000    0.000    0.000 internals.py:213(get_values)
            2    0.000    0.000    0.000    0.000 internals.py:233(mgr_locs)
            1    0.000    0.000    0.000    0.000 internals.py:3311(ndim)
            1    0.000    0.000    0.000    0.000 internals.py:3351(_is_single_block)
            1    0.000    0.000    0.000    0.000 internals.py:3384(_get_items)
            1    0.000    0.000    0.000    0.000 internals.py:3473(__len__)
            1    0.000    0.000    0.000    0.000 internals.py:3776(is_consolidated)
            1    0.000    0.000    0.000    0.000 internals.py:3922(as_array)
            1    0.000    0.000    0.000    0.000 internals.py:4085(consolidate)
           10    0.000    0.000    0.001    0.000 iostream.py:180(schedule)
            3    0.000    0.000    0.000    0.000 iostream.py:284(_is_master_process)
            3    0.000    0.000    0.000    0.000 iostream.py:297(_schedule_flush)
            2    0.000    0.000    0.005    0.002 iostream.py:311(flush)
            3    0.000    0.000    0.000    0.000 iostream.py:342(write)
           10    0.000    0.000    0.000    0.000 iostream.py:87(_event_pipe)
         2489    0.003    0.000    0.062    0.000 lapack.py:496(get_lapack_funcs)
         1990    0.003    0.000    0.003    0.000 linalg.py:111(get_linalg_error_extobj)
         1990    0.003    0.000    0.005    0.000 linalg.py:116(_makearray)
         3980    0.002    0.000    0.002    0.000 linalg.py:121(isComplexType)
         1990    0.001    0.000    0.002    0.000 linalg.py:134(_realType)
         1990    0.006    0.000    0.009    0.000 linalg.py:144(_commonType)
         1990    0.002    0.000    0.002    0.000 linalg.py:203(_assertRankAtLeast2)
         1990    0.002    0.000    0.002    0.000 linalg.py:209(_assertNdSquareness)
         1990    0.001    0.000    0.001    0.000 linalg.py:482(_unary_dispatcher)
         1990    0.065    0.000    0.092    0.000 linalg.py:486(inv)
         2489    0.001    0.000    0.001    0.000 misc.py:169(_datacopied)
            1    0.000    0.000    0.000    0.000 multiarray.py:145(concatenate)
         1989    0.001    0.000    0.001    0.000 multiarray.py:312(where)
         8456    0.002    0.000    0.002    0.000 multiarray.py:707(dot)
         1000    0.001    0.000    0.002    0.000 numerictypes.py:293(issubclass_)
          500    0.001    0.000    0.004    0.000 numerictypes.py:365(issubdtype)
         4978    0.003    0.000    0.004    0.000 numerictypes.py:578(_can_coerce_all)
         2489    0.007    0.000    0.013    0.000 numerictypes.py:602(find_common_type)
         2489    0.002    0.000    0.002    0.000 numerictypes.py:654(<listcomp>)
         2489    0.000    0.000    0.000    0.000 numerictypes.py:655(<listcomp>)
            1    0.000    0.000    0.000    0.000 os.py:664(__getitem__)
            1    0.000    0.000    0.000    0.000 os.py:742(encode)
            1    0.000    0.000    0.000    0.000 progressbar.py:131(__call__)
            1    0.000    0.000    0.000    0.000 progressbar.py:144(__iter__)
            2    0.000    0.000    0.005    0.003 progressbar.py:148(__next__)
            1    0.000    0.000    0.000    0.000 progressbar.py:168(_env_size)
            1    0.000    0.000    0.000    0.000 progressbar.py:174(_handle_resize)
            2    0.000    0.000    0.000    0.000 progressbar.py:181(percentage)
            2    0.000    0.000    0.000    0.000 progressbar.py:192(_format_widgets)
            2    0.000    0.000    0.000    0.000 progressbar.py:219(_format_line)
            2    0.000    0.000    0.000    0.000 progressbar.py:228(_need_update)
            1    0.000    0.000    0.000    0.000 progressbar.py:236(_update_widgets)
            4    0.000    0.000    0.000    0.000 progressbar.py:239(<genexpr>)
            2    0.000    0.000    0.005    0.003 progressbar.py:243(update)
            1    0.000    0.000    0.004    0.004 progressbar.py:267(start)
            1    0.000    0.000    0.002    0.002 progressbar.py:296(finish)
            1    0.000    0.000    0.000    0.000 progressbar.py:94(__init__)
           12    0.000    0.000    0.000    0.000 threading.py:1062(_wait_for_tstate_lock)
           12    0.000    0.000    0.000    0.000 threading.py:1104(is_alive)
            2    0.000    0.000    0.000    0.000 threading.py:215(__init__)
            2    0.000    0.000    0.000    0.000 threading.py:239(__enter__)
            2    0.000    0.000    0.000    0.000 threading.py:242(__exit__)
            2    0.000    0.000    0.000    0.000 threading.py:248(_release_save)
            2    0.000    0.000    0.000    0.000 threading.py:251(_acquire_restore)
            2    0.000    0.000    0.000    0.000 threading.py:254(_is_owned)
            2    0.000    0.000    0.004    0.002 threading.py:263(wait)
            2    0.000    0.000    0.000    0.000 threading.py:498(__init__)
           12    0.000    0.000    0.000    0.000 threading.py:506(is_set)
            2    0.000    0.000    0.004    0.002 threading.py:533(wait)
            2    0.000    0.000    0.000    0.000 twodim_base.py:154(eye)
         2489    0.001    0.000    0.001    0.000 type_check.py:209(_is_type_dispatcher)
         2489    0.003    0.000    0.004    0.000 type_check.py:282(iscomplexobj)
            1    0.089    0.089    0.847    0.847 utilities.py:179(dpmm_algorithm0)
          500    0.002    0.000    0.002    0.000 utilities.py:305(<listcomp>)
            2    0.000    0.000    0.000    0.000 widgets.py:229(update)
            2    0.000    0.000    0.000    0.000 widgets.py:299(update)
            8    0.000    0.000    0.000    0.000 widgets.py:302(<genexpr>)
           10    0.000    0.000    0.000    0.000 widgets.py:38(format_updatable)
            4    0.000    0.000    0.000    0.000 {built-in method _thread.allocate_lock}
         7467    0.007    0.000    0.007    0.000 {built-in method builtins.abs}
            1    0.000    0.000    0.000    0.000 {built-in method builtins.any}
            1    0.000    0.000    0.847    0.847 {built-in method builtins.exec}
         7472    0.008    0.000    0.008    0.000 {built-in method builtins.getattr}
           10    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        14944    0.006    0.000    0.006    0.000 {built-in method builtins.isinstance}
         9959    0.003    0.000    0.003    0.000 {built-in method builtins.issubclass}
            1    0.000    0.000    0.000    0.000 {built-in method builtins.iter}
    14958/14956    0.003    0.000    0.003    0.000 {built-in method builtins.len}
          503    0.002    0.000    0.002    0.000 {built-in method builtins.max}
            2    0.000    0.000    0.000    0.000 {built-in method builtins.next}
            1    0.000    0.000    0.000    0.000 {built-in method fcntl.ioctl}
            2    0.000    0.000    0.000    0.000 {built-in method math.ceil}
         2489    0.002    0.000    0.002    0.000 {built-in method math.log}
        17427    0.027    0.000    0.027    0.000 {built-in method numpy.array}
    27873/27870    0.044    0.000    0.281    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
            1    0.000    0.000    0.000    0.000 {built-in method numpy.core._multiarray_umath.normalize_axis_index}
          502    0.001    0.000    0.001    0.000 {built-in method numpy.empty}
            2    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
            3    0.000    0.000    0.000    0.000 {built-in method posix.getpid}
           10    0.000    0.000    0.000    0.000 {built-in method posix.urandom}
            3    0.000    0.000    0.000    0.000 {built-in method time.time}
         1990    0.000    0.000    0.000    0.000 {method '__array_prepare__' of 'numpy.ndarray' objects}
            2    0.000    0.000    0.000    0.000 {method '__enter__' of '_thread.lock' objects}
            2    0.000    0.000    0.000    0.000 {method '__exit__' of '_thread.lock' objects}
           20    0.004    0.000    0.004    0.000 {method 'acquire' of '_thread.lock' objects}
         2489    0.002    0.000    0.017    0.000 {method 'all' of 'numpy.ndarray' objects}
            2    0.000    0.000    0.000    0.000 {method 'append' of 'collections.deque' objects}
         2496    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
         1991    0.005    0.000    0.005    0.000 {method 'astype' of 'numpy.ndarray' objects}
          500    0.030    0.000    0.044    0.000 {method 'choice' of 'numpy.random.mtrand.RandomState' objects}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
            1    0.000    0.000    0.000    0.000 {method 'encode' of 'str' objects}
            1    0.000    0.000    0.000    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        10457    0.004    0.000    0.004    0.000 {method 'get' of 'dict' objects}
         2489    0.001    0.000    0.001    0.000 {method 'index' of 'list' objects}
            2    0.000    0.000    0.000    0.000 {method 'insert' of 'list' objects}
        12945    0.002    0.000    0.002    0.000 {method 'items' of 'dict' objects}
            2    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
          501    0.000    0.000    0.000    0.000 {method 'keys' of 'dict' objects}
            4    0.000    0.000    0.000    0.000 {method 'ljust' of 'str' objects}
         2489    0.001    0.000    0.001    0.000 {method 'lower' of 'str' objects}
            1    0.000    0.000    0.000    0.000 {method 'nonzero' of 'numpy.ndarray' objects}
            2    0.000    0.000    0.000    0.000 {method 'pop' of 'list' objects}
        15434    0.087    0.000    0.087    0.000 {method 'reduce' of 'numpy.ufunc' objects}
            2    0.000    0.000    0.000    0.000 {method 'release' of '_thread.lock' objects}
            1    0.000    0.000    0.000    0.000 {method 'sort' of 'numpy.ndarray' objects}
         2489    0.002    0.000    0.002    0.000 {method 'squeeze' of 'numpy.ndarray' objects}
            1    0.000    0.000    0.000    0.000 {method 'transpose' of 'numpy.ndarray' objects}
    
    



```python
# Affichage des threads sous forme de graphe

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

graphviz = GraphvizOutput()
graphviz.output_file = '../figures/threads_baseline_algorithm.png'

with PyCallGraph(output=graphviz, config=None):
    dpmm_algorithm0(data, alpha, mu_0, Sigma_0, sigma_x, 1, c_pooled)
```

    100% |########################################################################|



```python
from IPython.display import Image
Image(filename='../figures/threads_baseline_algorithm.png') 
```




![png](output_31_0.png)



Cette première version de l'algorithme ne fait pas appel à des fonctions intermédiaires. Pour optimiser les performances, nous allons réécrire cet algorithme et le tester dans le notebook suivant ; notamment en introduisant trois fonctions clefs dans le calcul et la performance. Dans le notebook 3, nous présenteront les résultats avec ces fonctions réécrites et dans le 4e notebook nous verrons les performances lorsqu'elles sont parallélisées.

### Conclusions

On observe que l'algorithme s'il parvient à prédire de manière très satisfaisante les clusters d'appartenance quand les données sont séparées comme c'est le cas ici, celui-ci peut être très lent dans son execution. Si l'on se refère au graphique ci avant on peut observer que bon nombres de fonction python sont appelées très souvent pouvant être une des causes de la lenteur de l'algorithme.
Afin de mieux comprendre la structure des appels des fonctions, nous allons procéder dans le prochain notebook à la fonctionnalisation de l'algorithme en créant des fonctions externes qu'appellera notre algorithme.
