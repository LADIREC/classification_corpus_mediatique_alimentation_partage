#################################################### Création des matrices d'occurrences pour l'entrainement du modèle de classification supervisée

# Note: la licence émise par Cision pour le moissonnage des données textuelles à travers l'interface Eureka ne permet pas le partage des données brutes.
# Le script ci-dessous ne peut donc être exécuté tel quel. Il est rendu public à titre de trace des opérations de prétraitement.

# Définir le chemin vers le répertoire de travail
# setwd('')

# Importation des extensions
libs <- c("dplyr", "stringr", "data.table", "text2vec", "lsa")
lapply(libs, require, character.only = TRUE)
set.seed(100)

# Nettoyage de l'environnement
rm(list = setdiff(ls(), c("recits_faim")))

# Importation des données
recits_faim <- readRDS('')


# Croisement des doc_idx avec ceux des documents annotés à la main par l'équipe
validation <- readxl::read_excel("") %>%
  setDT()
validation[TAGS %in% c("2keep", "2rmv"), `:=`(docID = str_extract(docID, "^[0-9]{10}"),
                                              TAGS = as.factor(TAGS))]

#Vérification
table(validation$TAGS)

validation <-
  merge(validation, recits_faim, by.x = "docID", by.y = "doc_id")
validation <-
  validation[TAGS %in% c("2keep", "2rmv"), .(docID, text, TAGS)]

# Observation des proportions
table(validation$TAGS)

setnames(validation, old = c("docID", "text", "TAGS"), new = c("ID", "U1", "IsToKeep"))

# Équilibrer le jeu de données
# # Générer des doublons dans docs2rmv pour atteindre le nombre de docs2keep
# # D'abord, conserver tous les doc_idx utilisés; ils serviront plus tard
doc_idx <- validation[, .(ID)]
n <-
  validation[IsToKeep == "2keep", .N] - validation[IsToKeep == "2rmv", .N]
doublons2rmv <-
  sample_n(validation[IsToKeep == "2rmv"], size = n, replace = FALSE)
validation <- bind_rows(validation, doublons2rmv)


# Mélanger ce jeu de données en vue de l'entrainement ultérieur
set.seed(123)
random_n <- runif(nrow(validation), min = 0)
data_binded <- validation[order(random_n), ]
data_binded[, IsToKeep := ifelse(IsToKeep == "2keep", "yes", "no")]



# saveRDS(data_binded, "data/20220606_databinded.RDS")
# saveRDS(doc_idx, "data/20220606_idx.RDS")
# # saveRDS(recits_faim, "data/20220301recits_clean.RDS")
# # data_binded <- readRDS("data/20211111_databinded.RDS")
# # recits_faim <- readRDS("data/recits_clean.RDS")

# Nettoyer l'environnement de travail
rm(list = setdiff(ls(), c("data_binded", "recits_faim", "doc_idx")))


# Création des matrices d'occurrences
setkey(data_binded, ID)
set.seed(234)
all_ids <- data_binded$ID
train_ids <- sample(all_ids, length(all_ids) * .80)
test_ids <- setdiff(all_ids, train_ids)
train = data_binded[J(train_ids)]
test = data_binded[J(test_ids)]

tok_fun = word_tokenizer

it_train <- itoken(
  train$U1,
  tokenizer = tok_fun,
  ids = train$ID,
  progressbar = FALSE
)

vocab = create_vocabulary(it_train)

vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)


# Nouveau vocabulaire écrémé
vocab_pruned <- create_vocabulary(it_train)
vocab_pruned <- prune_vocabulary(vocab_pruned,term_count_min = 25,
                                 doc_proportion_max = 0.6)
vectorizer_pruned <- vocab_vectorizer(vocab_pruned)

dtm_train_pruned <- create_dtm(it_train, vectorizer_pruned)
dim(dtm_train_pruned)

dtm_test_pruned <- create_dtm(it_test, vectorizer_pruned)# 
dim(dtm_test_pruned)

# # Transformation en matrice (il s'agit actuellement d'une matrice comprimée)
# dtm_train_pruned <- as.matrix(dtm_train_pruned)
# dtm_test_pruned <- as.matrix(dtm_test_pruned)
# 
# # Ajout des classes
# dtm_train_pruned <- ifelse(dtm_train_pruned>0,1,0)
# dtm_test_pruned <- ifelse(dtm_test_pruned>0,1,0)

# Transformation de la structure
dtm_train_pruned_df <- as.data.frame.matrix(dtm_train_pruned)
dtm_test_pruned_df <- as.data.frame.matrix(dtm_test_pruned)

# Ajout des classes
dtm_train_pruned_df$IsToKeep <- as.factor(train$IsToKeep)
dtm_test_pruned_df$IsToKeep <- as.factor(test$IsToKeep)

# Vérification que les n-grammes créés dans le premier script se trouvent dans la matrice d'occurrences
"insécurité_alimentaire" %in% names(dtm_test_pruned_df)

# # ==================================> Pondération TFIDF
vocab_tfidf <- create_vocabulary(it_train)
vocab_tfidf <-  prune_vocabulary(vocab_tfidf,
                                 term_count_min = 20,
                                 doc_proportion_max = 0.6)
vectorizer_tfidf <- vocab_vectorizer(vocab_tfidf)
dtm_train_tfidf <- create_dtm(it_train, vectorizer_tfidf)

tfidf = TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train_tfidf, tfidf)
it_test <- tok_fun(test$U1)
it_test <- itoken(it_test, ids = test$ID,
                  progressbar = FALSE)

dtm_test_tfidf <- create_dtm(it_test, vectorizer_tfidf)
dtm_test_tfidf <- transform(dtm_test_tfidf, tfidf)
dim(dtm_train_tfidf)
dim(dtm_test_tfidf)

dtm_train_tfidf_df <- as.data.frame.matrix(dtm_train_tfidf)
dtm_test_tfidf_df <- as.data.frame.matrix(dtm_test_tfidf)

dtm_train_tfidf_df$IsToKeep <- as.factor(train$IsToKeep)
dtm_test_tfidf_df$IsToKeep <- as.factor(test$IsToKeep)

saveRDS(dtm_train_tfidf_df, 'data/dtm_train_tfidf_df.RDS')
saveRDS(dtm_test_tfidf_df, 'data/dtm_test_tfidf_df.RDS')

dtm_train_tfidf_df <- readRDS('data/dtm_train_tfidf_df.RDS')
dtm_test_tfidf_df <- readRDS('data/dtm_test_tfidf_df.RDS')

dtm_all <- bind_rows(dtm_train_tfidf_df, dtm_test_tfidf_df)
saveRDS(dtm_all, "data/dtm_all_tfidf_df.RDS")

