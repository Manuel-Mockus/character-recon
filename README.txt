Le fichier mnist-original.mat est un fichier compatible avec Matlab et qui contient 70000 chiffres écrits à la main. Chaque chiffre est stockée comme un vecteur de taille 784 = 28*28

Les fichier dataset_*.mat sont des dubdivisions de la base de donees originale.

Le fichier functions.py contient les algorithmes implementés, aussi que des fonctions auxiliaires pour le rapport et la création de graphes recensant nos résultats.

run.py est un fichier pour tester les algorithmes. utilisation:

python3 run.py <nom de la base de donees> <Nb de l'lagorithme : 1 ou 2>

1 : Algorithme de clasification basé sur le calcul des images moyennes
2 : Algorithme basé sur la decomposition SVD
3 : Algorithme de la distance tangente

exemple:

python3 run.py dataset_Manuel.mat 2





