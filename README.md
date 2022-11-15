# Classification of text files from English and French news article corpora

## Auteurs et autrices:
* Développement corpus_fr: Pascal Brissette
* Développement corpus_en: Lisa Teichmann
* Classification manuelle: Alexia Wildhaber-Riley, Yu Chen Shi, Pascal Brissette
* Moissonnage du corpus recits_faims: Julien Vallières-Gingras

## Dossiers
corpus_fr contient les résultats et le code associés au corpus d'articles d'actualité en français

corpus_en contient les résultats et le code associés au corpus d'articles d'actualité en anglais

## Rapports
Chaque dossier comprend un rapport détaillé qui documente le modèle et les résultats du processus de classification en comparant différents algorithmes d'apprentissage automatique.

## Description
L’opération de classification consiste à sélectionner d’abord un échantillon de documents et à classifier ceux-ci manuellement selon des critères définis dans un guide de classification développé par le LADIREC. Selon ce guide, dans le corpus final, tous les documents font référence à au moins une des catégories suivantes:

1.	Production
*	Ex.: agriculture, serriculture, apiculture, élevage animal destiné à la consommation, production d'alcools et de boissons non alcoolisées, production de plats et d'autres produits destinés à l'alimentation, empaquetage
2.	Entreposage et distribution
*	Ex.: transport des aliments entre leurs lieux de production et leur point de vente ou de don, chaîne d’approvisionnement
3.	Vente au détail et consommation
*	Ex : accès aux marchés et aux banques de dons, marketing, prix de vente, restauration, bars, cafés
4.	Gestion des déchets
*	Ex. : compost, enfouissement, revalorisation
5.	Communications
*	Ex. : livres, conférences, comme sujet de débat politique

Les documents ne sont pas classifiés selon leur degré de pertinence: la présence d’au moins une expression ou d’un mot renvoyant à l'alimentation humaine suffit pour qu’un document soit considéré comme pertinent. Puis, nous avons formé un algorithme d’apprentissage automatique pour ne garder que les documents pertinents. Il est à noter que les documents écartés à cette étape ne sont pas détruits, mais pourraient servir à des analyses subséquentes.
