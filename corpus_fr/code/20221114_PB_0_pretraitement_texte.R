#################################################### Prétraitement du corpus

# Note: la licence émise par Cision pour le moissonnage des données textuelles à travers l'interface Eureka ne permet pas le partage des données brutes.
# Le script ci-dessous ne peut donc être exécuté tel quel. Il est rendu public à titre de trace des opérations de prétraitement.

# Définir le chemin vers votre répertoire de travail
# setwd('')

libs <- c( "stringr", "dplyr", "tidytext", "lsa", "tm", "data.table")
lapply(libs, require, character.only = TRUE)
rm(list = ls())
gc()


# ================================================ Importation du csv initial
# recits_faim <- fread('')

setnames(recits_faim, c("U1", "ID"), c("text", "doc_id")) # Identifier les colonnes texte et doc_id

setcolorder(recits_faim, c("doc_id", "text")) # Placer les deux colonnes importantes au début du tableau

# str(recits_faim) # Observer le résultat

# ================================================ Traitement des cellules de la colonne "text" sans contenu
# Calcul du nombre de cellules sans contenu
recits_faim[text == "null",.N]

# Si la colonne text est sans contenu, on lui fournit celui d'une autre colonne
recits_faim[, text := ifelse(text == "null", U2, text)]
recits_faim[, text := ifelse(text == "null", U3, text)]
recits_faim[, text := ifelse(text == "null", L4, text)]

recits_faim[text == "null",.N]

# Élimination du document restant sans contenu textuel
recits_faim <- recits_faim[ text != "null"]

# L'opération ci-dessus a généré des "textes" parfois trop courts pour être significatifs
# Élimination des documents dont les textes comortent moins de 85 caractères
recits_faim <- recits_faim[ nchar(text) > 85] # Cette simple ligne remplace le code ci-dessous


# ================================================ Filtrage des documents par langue =========> La suite des opérations n'est faite que sur les documents en français

# Filtrage des documents par langue
recits_faim <- recits_faim[LA == "Français"]

recits_faim <- recits_faim[, .(doc_id, text, AU, TI, JF, CY, OP, PY, U4, TY)]

# attribution de nouveaux noms aux colonnes conservées
setnames(recits_faim, 
         c("doc_id", "text", "AU", "TI", "JF", "CY", "OP", "PY", "U4", "TY"),
         c("doc_id", "text", "auteur", "titre", "media", "lieu", "date_pub_or", "annee", "source", "type")
)
# str(recits_faim) # Vérification

# Supprimer les doublons
# recits_faim <- unique(recits_faim, by = "titre")
recits_faim <- unique(recits_faim, by = "text")

# saveRDS(recits_faim, 'data/20221114_PB_recits_raw_sans_doublon.RDS')
# recits_faim <- readRDS('data/20221114_PB_recits_raw_sans_doublon.RDS')

# Réduction de la structure
recits_faim <- recits_faim[, .(doc_id, text)]

# ================================================ Élimination des symboles
recits_faim[4,2]
recits_faim[, text := gsub("[^[:alnum:]]", " ", text)]
# recits_faim[, text := gsub("’", "'", text)]
# recits_faim[, text := gsub('["«»><]', ' ', text)]
# recits_faim[, text := str_replace_all(text, '\\s(?=%)', '')]
# recits_faim[, text := gsub('\\', '', text, fixed = TRUE)]
# recits_faim[, text := gsub('""""""""', '', text, fixed = TRUE)]
recits_faim[, text := stripWhitespace(text)]
recits_faim[, text := tolower(text)]


# # ======================================================== Application d'un antidictionnaire
# 
lsastops <- list(mots = lsa::stopwords_fr)
setDT(lsastops)

# Révision de l'antidictionnaire lsa. Retrait de 107 mots de cette liste.
lsastops <- lsastops[!mots %in% c(
  "ailleurs","alentour","alias","après","après-demain","arrière","attendu","aujourd",
  "aujourdhui","auparavant","auprès","aussitôt","autrefois","autres","autrui","avant-hier",
  "bientôt", "beaucoup","céans","chiche","davantage","demain","dedans","dehors",
  "derrière","désormais","différent","différente","différentes","différents",
  "divers","diverses","dorénavant","entre-temps","environ","excepté","exprès","extenso",
  "extremis","fi","fortes","grosso","guère","haut","hélas","holà","ici","ici-bas","importe",
  "ipso","item","jamais","juste","loin","longtemps","maint","mainte","maints",
  "maintes","maintenant","malgré","même","mêmes","mille","milliards","millions",
  "mince","minima","modo","moi","moult","moyennant","naguère","nombreux","nombreuses",
  "nul","nulle","olé","ollé","parbleu","particulier","particulière","particulièrement","partout",
  "passé","personne","posteriori","pourquoi","premier","préalable","proche","sacrebleu",
  "soudain","stop","stricto","sur-le-champ","tard","tenant","toujours","touchant","vif",
  "vifs","vitro","vite","volontiers","zéro","zut"
)]

# Utilise tidytext pour supprimer les mots fonctionnels de l'antidictionnaire
antidictionnaire_fr <- tibble(word = lsastops[,mots])
recits_copie <- recits_faim
recits_copie <- recits_copie %>% unnest_tokens(output = "word",
                                               input = "text",
                                               token = "words")
recits_copie <- recits_copie %>% anti_join(antidictionnaire_fr, by = "word")
recits_copie <- recits_copie %>% group_by(doc_id) %>% summarise(text=paste0(word, collapse = " "))

recits_faim <- arrange(recits_faim, doc_id)
recits_copie <- arrange(recits_copie, doc_id)

recits_faim$text <- recits_copie$text

recits_faim$text[1]
setDT(recits_faim)

# ======================================================== Ajout de mots composés

# Importation d'une liste des noms de quartiers composés accompagnés de regex
quartiers <- fread('data/quartierreferencehabitation.csv')

# Importation d'une liste des noms d'organismes
organismes <- fread('data/organismes.csv')

# Importation d'une liste de mots composés fréquents
divers <- fread('data/divers.csv')

# Composition des noms de quartiers
for(i in 1:nrow(quartiers)){
  recits_faim$text <- recits_faim$text %>% str_replace_all(quartiers$a[i], quartiers$b[i])
}
# Vérification
recits_faim[text %like% "rivière_des_prairies", .N]

# Composition des noms d'organismes
for(i in 1:nrow(organismes)) {
  recits_faim$text <- recits_faim$text %>% str_replace_all(organismes$a[i], organismes$b[i])
}

# Vérification
recits_faim[text %like% "fondation_chagnon", .N]

# Composition des noms d'unités lexicales multiples diverses
for(i in 1:nrow(divers)) {
  recits_faim$text <- recits_faim$text %>% str_replace_all(divers$a[i], divers$b[i])
}

# Vérification
recits_faim[text %like% "aliment_transformé", .N]

rm(list = setdiff(ls(), c("recits_faim")))

# Sauvegarder la structure de données
saveRDS(recits_faim, "data/20221114_PB_recits_net.RDS")
# recits_faim <- readRDS('data/recits_faim_net.RDS')

# ================================================ Nettoyage et traitement des TITRES par antidictionnaire

recits_faim[, titre := tolower(titre)]

recits_faim[, titre := lapply(titre, str_replace_all, '[^[:alnum:]]|\\b[:alpha:]{1,2}\\b|[:digit:]{5,}', ' ')]

# Utilise tidytext pour supprimer les mots fonctionnels
recits_copie <- recits_faim
recits_copie <- recits_copie %>% unnest_tokens(output = "word",
                                               input = "titre",
                                               token = "words")
recits_copie <- recits_copie %>% anti_join(antidictionnaire_fr, by = "word")
recits_copie <- recits_copie %>% group_by(doc_id) %>% summarise(titre=paste0(word, collapse = " "))

recits_faim <- arrange(recits_faim, doc_id)
recits_copie <- arrange(recits_copie, doc_id)

recits_faim <- left_join(recits_copie, recits_faim, by = "doc_id")
identical(recits_faim$doc_id, recits_copie$doc_id)

recits_faim$titre <- recits_copie$titre

rm(recits_copie, lsastops, antidictionnaire_fr)

setDT(recits_faim)
recits_faim[, c("titre.x", "titre.y") := NULL]
setcolorder(recits_faim, c("doc_id", "text", "titre", "auteur"))

recits_faim[, titre := stripWhitespace(titre)]
recits_faim[, titre := str_trim(titre)]

# Composition des noms de quartier
for(i in 1:nrow(quartiers)){
  recits_faim$titre <- recits_faim$titre %>% str_replace_all(quartiers$a[i], quartiers$b[i])
}

# Vérification
nrow(filter(recits_faim, str_detect(titre, "quartier_latin")))

# Composition des noms d'organismes
for(i in 1:nrow(organismes)) {
  recits_faim$titre <- recits_faim$titre %>% str_replace_all(organismes$a[i], organismes$b[i])
}

# Vérification
nrow(filter(recits_faim, str_detect(titre, "moisson_montréal")))

# Composition des noms d'unités lexicales multiples diverses
for(i in 1:nrow(divers)) {
  recits_faim$titre <- recits_faim$titre %>% str_replace_all(divers$a[i], divers$b[i])
}

# Vérification
nrow(filter(recits_faim, str_detect(titre, "sirop_érable")))

rm(list = setdiff(ls(), c("recits_faim")))
