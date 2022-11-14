# Classification of text files from English and French news article corpora

## Dossiers
corpus_en contains all data and code associated with the corpus of English news articles

corpus_fr contains all data and code associated with the corpus of French news articles

## Rapports
Chaque dossier comprend un rapport détaillé qui documente le modèle et nos résultats du processus de classification en comparant différents algorithmes d'apprentissage automatique.

## Description
L’opération de classification consiste à sélectionner d’abord manuellement les documents pertinents du corpus selon des critères définies dans un guide de classification développé par le LADIREC. Selon ce guide, dans le corpus final, tous les documents font référence à au moins une des catégories suivantes:

a.	Production
i.	Ex.: agriculture, serriculture, apiculture, élevage animal destiné à la consommation, production d'alcools et de boissons non alcoolisées, production de plats et d'autres produits destinés à l'alimentation, empaquetage
b.	Entreposage et distribution
i.	Ex.: transport des aliments entre leurs lieux de production et leur point de vente ou de don, chaîne d’approvisionnement
c.	Vente au détail et consommation
i.	Ex : accès aux marchés et aux banques de dons, marketing, prix de vente, restauration, bars, cafés
d.	Gestion des déchets
i.	Ex. : compost, enfouissement, revalorisation
e.	Communications
i.	Ex. : livres, conférences, comme sujet de débat politique

 Les documents ne sont pas classifiés selon leur degré de pertinence: la présence d’au moins une expression ou d’un mot pertinent à la question de recherche suffit pour qu’un document soit considéré comme pertinent. Puis, nous avons formé un algorithme d’apprentissage automatique pour ne garder que les documents pertinents. Il est à noter que les documents écartés à cette étape ne sont pas détruits, mais pourraient servir à des analyses subséquentes.
