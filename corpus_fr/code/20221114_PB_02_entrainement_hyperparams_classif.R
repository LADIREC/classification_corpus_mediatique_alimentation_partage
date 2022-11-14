#################################################### Entrainement de plusieurs modèles, repérage des meilleurs hyperparamètres du meilleur modèle, classification à l'aveugle

# Note: la licence émise par Cision pour le moissonnage des données textuelles à travers l'interface Eureka ne permet pas le partage des données brutes.
# Le script ci-dessous ne peut donc être exécuté tel quel. Il est rendu public à titre de trace des opérations de prétraitement.

# Définir le chemin vers le répertoire de travail
# setwd('')

# Classification avec Random Forest
# Le jeu de données importé a été allégé (term_count_min = 20,doc_proportion_max = 0.5).
# Une sélection des plus importants éléments (features) sera fait en utilisant la mesure MeanDecreaseGini>0.5.
library(randomForest)
library(caret)
library(data.table)
library(text2vec)
library(doParallel)
library(foreach)
library(readr)


# Chargement des extensions et des données prétraitées
# setwd('')

dtm_train_tfidf_df <- readRDS('')
dtm_test_tfidf_df <- readRDS('')
dtm_all <- readRDS('')

# # Ce premier entrainement vise à extraire les éléments (features) les plus importants et à supprimer le bruit

classifier_tfidf <- randomForest(x = dtm_all[, -ncol(dtm_all)],
                                 y = dtm_all[, ncol(dtm_all)],
                                 # mtry = 58,
                                 # ntree = 500,
                                 importance = TRUE)

classifier_tfidf <- readRDS("data/20220613_PB_classifier_imp.RDS")
# ============================================ Sélection des éléments les plus importants (meilleur résultat obtenu: Mean Accurracy92.37%)
# Pruning the set with the MeanDecreaseGini
importance <- randomForest::importance(classifier_tfidf, type = 2)


my_df <- data.table(variable = attributes(importance)$dimnames[[1]],
                    valeur = importance)



# =========================================== Tester différents seuils Gini

meanDecrGini <-
  c(
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.90,
    0.95,
    1,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.3,
    1.35,
    1.4
  )
length(meanDecrGini)
featureSelect <- data.table(
  meanDecrGini = double(),
  Kappa = double(),
  Sensitivity = double(),
  Specificity = double(),
  OverallAccuracy = double()
)


featureSelections <- list()
for (i in seq_along(meanDecrGini)) {
  featureSelections[i] <-
    my_df[my_df$valeur.MeanDecreaseGini > meanDecrGini[i], 1]
}

set.seed(123)
require(tidyverse)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)
start.time <- proc.time()
for (i in seq_along(featureSelections)) {
  dtm_train_tfidf_imp_df <-
    dtm_train_tfidf_df[, colnames(dtm_train_tfidf_df) %in% featureSelections[[i]]]
  dtm_train_tfidf_imp_df$IsToKeep <- dtm_train_tfidf_df$IsToKeep
  dtm_test_tfidf_imp_df <-
    dtm_test_tfidf_df[, colnames(dtm_test_tfidf_df) %in% featureSelections[[i]]]
  dtm_test_tfidf_imp_df$IsToKeep <- dtm_test_tfidf_df$IsToKeep
  classifier_imp <-
    randomForest(x = dtm_train_tfidf_imp_df[,-ncol(dtm_train_tfidf_imp_df)],
                 y = dtm_train_tfidf_imp_df[, ncol(dtm_train_tfidf_imp_df)],
                 ntree = 500)
  predict_imp <-
    predict(classifier_imp, newdata = dtm_test_tfidf_imp_df[,-ncol(dtm_test_tfidf_imp_df)])
  cm_imp <- table(dtm_test_tfidf_imp_df$IsToKeep,
                  predict_imp,
                  dnn = c("Actual", "Predicted"))
  cm_imp_list <- confusionMatrix(cm_imp)
  featureSelectTemp <- data.table(
    meanDecrGini = meanDecrGini[[i]],
    Kappa = cm_imp_list$overall[[2]],
    Sensitivity = cm_imp_list$byClass[[1]],
    Specificity = cm_imp_list$byClass[[2]],
    OverallAccuracy = cm_imp_list$overall[[1]]
  )
  featureSelect <- bind_rows(featureSelect, featureSelectTemp)
}
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)

for (i in 1:nrow(featureSelect)) {
  featureSelect$n_features[i] <-
    nrow(my_df[my_df$valeur.MeanDecreaseGini > featureSelect$meanDecrGini[i], 1])
}


featureSelect <- featureSelect[, c(1, 6, 2, 3, 4, 5)]
fwrite(featureSelect, "results/FeatureSelectionScore.csv")
featureSelect <- fread("results/FeatureSelectionScore.csv")

bestMeanDecreaseGini <-
  featureSelect[featureSelect$OverallAccuracy == max(featureSelect$OverallAccuracy), 1]



# Visualisation des résultats

jpeg("results/20220620_PB_bestMeanDecreaseGine_plot.jpeg",
     quality = 600)

ggplot(featureSelect, aes(x = n_features, y = OverallAccuracy)) +
  geom_jitter() +
  geom_smooth(
    method = "lm",
    formula = y ~ x + I(x ^ 2),
    linetype = "dotted",
    se = FALSE
  ) +
  xlab("Nombre d'éléments") +
  ylab("Précision globale")


ggsave("results/20220620_PB_bestMeanDecreaseGine_plot.png", dpi = 300)

saveRDS(classifier_imp, "data/20220613_PB_classifier_imp.RDS")
classifier_tfidf <- readRDS("data/20220613_PB_classifier_imp.RDS")


variables2keep <-
  my_df[my_df$valeur.MeanDecreaseGini > bestMeanDecreaseGini$meanDecrGini, 1]

dtm_train_tfidf_imp_df <-
  dtm_train_tfidf_df[, colnames(dtm_train_tfidf_df) %in% variables2keep$variable]
dtm_train_tfidf_imp_df$IsToKeep <- dtm_train_tfidf_df$IsToKeep
dtm_test_tfidf_imp_df <-
  dtm_test_tfidf_df[, colnames(dtm_test_tfidf_df) %in% variables2keep$variable]
dtm_test_tfidf_imp_df$IsToKeep <- dtm_test_tfidf_df$IsToKeep

dim(dtm_train_tfidf_imp_df)
dim(dtm_test_tfidf_imp_df)

# ============================================= Comparaison de méthodes

names(dtm_train_tfidf_imp_df) <-
  janitor::make_clean_names(colnames(dtm_train_tfidf_imp_df))
names(dtm_test_tfidf_imp_df) <-
  janitor::make_clean_names(colnames(dtm_test_tfidf_imp_df))

saveRDS(dtm_train_tfidf_imp_df, "data/dtm_train_tfidf_imp_df.RDS")
saveRDS(dtm_test_tfidf_imp_df, "data/dtm_test_tfidf_imp_df.RDS")

dtm_train_tfidf_imp_df <- readRDS("data/dtm_train_tfidf_imp_df.RDS")
dtm_test_tfidf_imp_df <- readRDS("data/dtm_test_tfidf_imp_df.RDS")
dim(dtm_train_tfidf_imp_df)


### ============================================ Entrainement de différents modèles pour trouver le meilleur

# Définir des paramètres d'entrainement communs
require(mlbench)
require(e1071)


control <-
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 3)

set.seed(123)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)
start.time <- proc.time()
# CART
fit.cart <-
  train(is_to_keep ~ .,
        data = dtm_train_tfidf_imp_df,
        method = "rpart",
        trControl = control)

# LDA
fit.lda <-
  train(is_to_keep ~ .,
        data = dtm_train_tfidf_imp_df,
        method = "lda",
        trControl = control)

# SVM
fit.svm <-
  train(is_to_keep ~ .,
        data = dtm_train_tfidf_imp_df,
        method = "svmRadial",
        trControl = control)

# KNN
fit.knn <-
  train(is_to_keep ~ .,
        data = dtm_train_tfidf_imp_df,
        method = "knn",
        trControl = control)

# Random Forest
fit.rf <-
  train(is_to_keep ~ .,
        data = dtm_train_tfidf_imp_df,
        method = "rf",
        trControl = control)

results <-
  resamples(list(
    CART = fit.cart,
    LDA = fit.lda,
    SVM = fit.svm,
    KNN = fit.knn,
    RF = fit.rf
  ))

stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)


# Synthétiser les résultats
comparaison_modeles <- summary(results)
comparaison_modeles <-
  as.data.frame(comparaison_modeles$statistics$Accuracy)
comparaison_modeles$model <- rownames(comparaison_modeles)
setDT(comparaison_modeles)
comparaison_modeles[, `:=`(
  algorithm = model,
  min = Min.,
  median = Median,
  mean = Mean,
  max = Max.,
  Min. = NULL,
  `1st Qu.` = NULL,
  Median = NULL,
  Mean = NULL,
  `3rd Qu.` = NULL,
  Max. = NULL,
  `NA's` = NULL,
  model = NULL
)]
melt(comparaison_modeles)
ggplot(melt(comparaison_modeles), aes(x = reorder(algorithm, value), y =
                                        value)) +
  geom_boxplot() +
  ylab("Accuracy") +
  xlab("Algorithm")

comparaison_modeles_reshape <-
  results$values[, c("Min.", "Median", "Mean", "Max.")]
saveRDS(comparaison_modeles,
        'results/20220620_PB_ComparedModels_corpfr.RDS')
comparaison_modeles <-
  readRDS('results/20220620_PB_ComparedModels_corpfr.RDS')

ggsave("results/20220620_PB_ComparedModels_corpfr.png", dpi = 300)

# ========================================== Construction d'une fonction qui permette de repérer les meilleurs hyperparamètres de RF (mtry et ntrees)

customRF <-
  list(type = "Classification",
       library = "randomForest",
       loop = NULL)
customRF$parameters <-
  data.frame(
    parameter = c("mtry", "ntree"),
    class = rep("numeric", 2),
    label = c("mtry", "ntree")
  )
customRF$grid <- function(x, y, len = NULL, search = "grid") {
}
customRF$fit <-
  function(x,
           y,
           wts,
           param,
           lev,
           last,
           weights,
           classProbs,
           ...) {
    randomForest(x, y, mtry = param$mtry, ntree = param$ntree, ...)
  }
customRF$predict <-
  function(modelFit,
           newdata,
           preProc = NULL,
           submodels = NULL)
    predict(modelFit, newdata)
customRF$prob <-
  function(modelFit,
           newdata,
           preProc = NULL,
           submodels = NULL)
    predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x)
  x[order(x[, 1]), ]
customRF$levels <- function(x)
  x$classes


# Entrainement du modèle


set.seed(123)
trainControl <-
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 3)

custom_fun <- function(i) {
  train(
    is_to_keep ~ .,
    data = dtm_train_tfidf_imp_df,
    method = customRF,
    metric = "Accuracy",
    tuneGrid = expand.grid(.mtry = i, .ntree = 500),
    trControl = trainControl
  )
}

# Mélange des paramètres: Mtry entre 1 et 60, ntrees =500
numCores <- detectCores() - 1
stopImplicitCluster()
registerDoParallel(numCores)

cl <- makeCluster(numCores, type = "FORK")
start_time <- Sys.time()
customNtrees500 <- foreach(i = 1:50) %dopar% {
  custom_fun(i)
}
stopCluster(cl)
stop_time <- Sys.time()
duree_clusterNtrees500 <- stop_time - start_time

saveRDS(customNtrees500, 'data/customNtrees500.RDS')
customNtrees500 <- readRDS('data/customNtrees500.RDS')

#Extraction des performances pour mtry 1:60, ntrees=500
x <- character()
performances500 <- list()

for (i in seq_along(customNtrees500)) {
  x <- customNtrees500[[i]]$results
  performances500 <- rbind(x, performances500)
}
performances500

ggplot(performances500, aes(x = mtry, y = Accuracy)) +
  geom_point() +
  geom_line()
custom[[1]]$results


# Mtry entre 1 et 60, ntrees =1000
custom_fun <- function(i) {
  train(
    is_to_keep ~ .,
    data = dtm_train_tfidf_imp_df,
    method = customRF,
    metric = "Accuracy",
    tuneGrid = expand.grid(.mtry = i, .ntree = 1000),
    trControl = trainControl
  )
}
numCores <- detectCores() - 1
stopImplicitCluster()
registerDoParallel(numCores)

cl <- makeCluster(numCores, type = "FORK")
start_time <- Sys.time()
customNtrees1000 <- foreach(i = 1:50) %dopar% {
  custom_fun(i)
}
stopCluster()
stop_time <- Sys.time()
duree_clusterNtrees1000 <- stop_time - start_time
saveRDS(customNtrees1000, 'data/customNtrees1000.RDS')
#
x <- character()
performances1000 <- list()

for (i in seq_along(customNtrees1000)) {
  x <- customNtrees1000[[i]]$results
  performances1000 <- rbind(x, performances1000)
}
performances1000


# Mtry entre 1 et 60, ntrees =1500
custom_fun <- function(i) {
  train(
    is_to_keep ~ .,
    data = dtm_train_tfidf_imp_df,
    method = customRF,
    metric = "Accuracy",
    tuneGrid = expand.grid(.mtry = i, .ntree = 1500),
    trControl = trainControl
  )
}
numCores <- detectCores() - 1
stopImplicitCluster()
registerDoParallel(numCores)

cl <- makeCluster(numCores, type = "FORK")
start_time <- Sys.time()
customNtrees1500 <- foreach(i = 1:50) %dopar% {
  custom_fun(i)
}
stopCluster()
stop_time <- Sys.time()
duree_clusterNtrees1500 <- stop_time - start_time
saveRDS(customNtrees1500, 'data/customNtrees1500.RDS')


x <- character()
performances1500 <- list()

for (i in seq_along(customNtrees1500)) {
  x <- customNtrees1500[[i]]$results
  performances1500 <- rbind(x, performances1500)
}
performances1500

# Meilleures performances
summary(performances1500$Accuracy)
summary(performances1000$Accuracy)
summary(performances500$Accuracy)
# ggplot(performances1500, aes(x=mtry, y=Accuracy))+
#   geom_point()+
#   geom_line()
# custom[[1]]$results



#===================================> Compilation des résultats

customNtrees500 <- readRDS('data/customNtrees500.RDS')
customNtrees1000 <- readRDS('data/customNtrees1000.RDS')
customNtrees1500 <- readRDS('data/customNtrees1500.RDS')

extract_performance_fun <- function(z) {
  x <- character()
  performancesZ <- list()
  for (i in seq_along(z)) {
    x <- z[[i]]$results
    performancesZ <- rbind(x, performancesZ)
  }
  return(performancesZ)
}

combined_results <- rbind(
  extract_performance_fun(customNtrees500),
  extract_performance_fun(customNtrees1000),
  extract_performance_fun(customNtrees1500)
)


setDT(combined_results)
combined_results[order(Accuracy, decreasing = TRUE)]

fwrite(
  combined_results,
  "results/20220620_PB_performance_comparee_params_RF_corpfr.csv"
)
combined_results <-
  fread("results/20220620_PB_performance_comparee_params_RF_corpfr.csv")

# Visualisation des résultats
setDT(combined_results)
combined_results[, ntree := as.factor(ntree)]
levels(combined_results$ntree) <-
  c("ntree_500", "ntree_1000", "ntree_1500")
test_accuracy_plot <-
  ggplot(combined_results, aes(x = mtry, y = Accuracy)) +
  geom_point() +
  facet_wrap( ~ ntree) +
  ggtitle("Performance comparée de l'algorithme RF",
          subtitle = "Croisement des paramètres mtry et ntrees")

test_accuracy_plot <-
  ggplot(combined_results, aes(x = mtry, y = Accuracy, color = ntree)) +
  geom_point() +
  ggtitle("Performance comparée de l'algorithme RF",
          subtitle = "Croisement des paramètres mtry et ntrees")

ggsave("results/20220620_PB_performances_comparees_paramsRF_corpusfr.png",
       dpi = 300)


# =======================================> Créer des classifieurs avec les 5 meilleurs combinaisons

hyperParams_fun <- function(mtry, ntree) {
  randomForest(
    x = dtm_train_tfidf_imp_df[,-ncol(dtm_train_tfidf_imp_df)],
    y = dtm_train_tfidf_imp_df[, ncol(dtm_train_tfidf_imp_df)],
    mtry = mtry,
    ntree = ntree
  )
}

combined_results[order(Accuracy, decreasing = TRUE)][1:5]

set.seed(123)
cl <- makePSOCKcluster(9)
registerDoParallel(cl)
start.time <- proc.time()
classifier_47_500 <- hyperParams_fun(mtry = 47, ntree = 500)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)


set.seed(123)
cl <- makePSOCKcluster(9)
registerDoParallel(cl)
start.time <- proc.time()
classifier_39_500 <- hyperParams_fun(mtry = 39, ntree = 500)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)

set.seed(123)
cl <- makePSOCKcluster(9)
registerDoParallel(cl)
start.time <- proc.time()
classifier_34_1500 <- hyperParams_fun(mtry = 34, ntree = 1500)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)


set.seed(123)
cl <- makePSOCKcluster(9)
registerDoParallel(cl)
start.time <- proc.time()
classifier_39_1000 <- hyperParams_fun(mtry = 39, ntree = 1000)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)

set.seed(123)
cl <- makePSOCKcluster(9)
registerDoParallel(cl)
start.time <- proc.time()
classifier_40_1000 <- hyperParams_fun(mtry = 40, ntree = 1000)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
stopCluster(cl)


predict_47_500 <-
  predict(classifier_47_500, newdata = dtm_test_tfidf_imp_df[, -ncol(dtm_test_tfidf_imp_df)])
cm_imp_47_500 <-
  table(dtm_test_tfidf_imp_df$is_to_keep,
        predict_47_500,
        dnn = c("Actual", "Predicted"))
predict_39_500 <-
  predict(classifier_39_500, newdata = dtm_test_tfidf_imp_df[, -ncol(dtm_test_tfidf_imp_df)])
cm_imp_39_500 <-
  table(dtm_test_tfidf_imp_df$is_to_keep,
        predict_39_500,
        dnn = c("Actual", "Predicted"))
predict_34_1500 <-
  predict(classifier_34_1500, newdata = dtm_test_tfidf_imp_df[, -ncol(dtm_test_tfidf_imp_df)])
cm_imp_34_1500 <-
  table(dtm_test_tfidf_imp_df$is_to_keep,
        predict_34_1500,
        dnn = c("Actual", "Predicted"))
predict_39_1000 <-
  predict(classifier_39_1000, newdata = dtm_test_tfidf_imp_df[, -ncol(dtm_test_tfidf_imp_df)])
cm_imp_39_1000 <-
  table(dtm_test_tfidf_imp_df$is_to_keep,
        predict_39_1000,
        dnn = c("Actual", "Predicted"))
predict_40_1000 <-
  predict(classifier_40_1000, newdata = dtm_test_tfidf_imp_df[, -ncol(dtm_test_tfidf_imp_df)])
cm_imp_40_1000 <-
  table(dtm_test_tfidf_imp_df$is_to_keep,
        predict_40_1000,
        dnn = c("Actual", "Predicted"))


confusion_matrix_fin <- confusionMatrix(cm_imp_34_1500)
saveRDS(confusion_matrix_fin,
        "results/20220618_PB_confusionMatrixClassFin_fr.RDS")

classifier_final <- classifier_34_1500
saveRDS(classifier_final, "data/classifier_final.RDS")

# Fonction créée par "cybernetic": https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package
draw_confusion_matrix <- function(cm) {
  layout(matrix(c(1, 1, 2)))
  par(mar = c(2, 2, 2, 2))
  plot(
    c(100, 345),
    c(300, 450),
    type = "n",
    xlab = "",
    ylab = "",
    xaxt = 'n',
    yaxt = 'n'
  )
  title('CONFUSION MATRIX', cex.main = 2)
  
  
  # create the matrix
  rect(150, 430, 240, 370, col = '#3F97D0')
  text(195, 435, 'Class1', cex = 1.2)
  rect(250, 430, 340, 370, col = '#F7AD50')
  text(295, 435, 'Class2', cex = 1.2)
  text(125,
       370,
       'Predicted',
       cex = 1.3,
       srt = 90,
       font = 2)
  text(245, 450, 'Actual', cex = 1.3, font = 2)
  rect(150, 305, 240, 365, col = '#F7AD50')
  rect(250, 305, 340, 365, col = '#3F97D0')
  text(140, 400, 'Class1', cex = 1.2, srt = 90)
  text(140, 335, 'Class2', cex = 1.2, srt = 90)
  
  
  # add in the cm results
  res <- as.numeric(cm$table)
  text(195,
       400,
       res[1],
       cex = 1.6,
       font = 2,
       col = 'white')
  text(195,
       335,
       res[2],
       cex = 1.6,
       font = 2,
       col = 'white')
  text(295,
       400,
       res[3],
       cex = 1.6,
       font = 2,
       col = 'white')
  text(295,
       335,
       res[4],
       cex = 1.6,
       font = 2,
       col = 'white')
  
  # add in the specifics
  plot(
    c(100, 0),
    c(100, 0),
    type = "n",
    xlab = "",
    ylab = "",
    main = "DETAILS",
    xaxt = 'n',
    yaxt = 'n'
  )
  text(10, 85, names(cm$byClass[1]), cex = 1.2, font = 2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex = 1.2)
  text(30, 85, names(cm$byClass[2]), cex = 1.2, font = 2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex = 1.2)
  text(50, 85, names(cm$byClass[5]), cex = 1.2, font = 2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex = 1.2)
  text(70, 85, names(cm$byClass[6]), cex = 1.2, font = 2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex = 1.2)
  text(90, 85, names(cm$byClass[7]), cex = 1.2, font = 2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex = 1.2)
  
  
  # add in the accuracy information
  text(30, 35, names(cm$overall[1]), cex = 1.5, font = 2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex = 1.4)
  text(70, 35, names(cm$overall[2]), cex = 1.5, font = 2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex = 1.4)
}

draw_confusion_matrix(confusion_matrix_fin)



# =====================================> Classification des documents avec classifieur final
recits_faim <- readRDS('')
doc_idx <- readRDS('')

# Retirer du lot de documents ceux qui ont servi à l'entrainement
recits_faim <- recits_faim[!doc_id %in% doc_idx$ID]


data_binded <- readRDS("")

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

vocab_tfidf <- create_vocabulary(it_train)
vocab_tfidf <-  prune_vocabulary(vocab_tfidf,
                                 term_count_min = 20,
                                 doc_proportion_max = 0.6)
vectorizer_tfidf <- vocab_vectorizer(vocab_tfidf)
dtm_train_tfidf <- create_dtm(it_train, vectorizer_tfidf)

tfidf = TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train_tfidf, tfidf)
it_final <- tok_fun(recits_faim$text)
it_final <- itoken(it_final, ids = recits_faim$doc_id,
                   progressbar = FALSE)


dtm_final_tfidf <- create_dtm(it_final, vectorizer_tfidf)
dtm_final_tfidf <- transform(dtm_final_tfidf, tfidf)
dim(dtm_train_tfidf)
dim(dtm_final_tfidf)

dtm_final_tfidf_df <- as.data.frame.matrix(dtm_final_tfidf)
names(dtm_final_tfidf_df) <-
  janitor::make_clean_names(colnames(dtm_final_tfidf_df))
setDT(dtm_final_tfidf_df)

myVariables <-
  colnames(dtm_train_tfidf_imp_df[,-ncol(dtm_train_tfidf_imp_df)])

dtm_final_tfidf_df <- dtm_final_tfidf_df[, ..myVariables]

predict_final <-
  predict(classifier_final, newdata = dtm_final_tfidf_df)

recits_faim$IsToKeep <- predict_final
recits_faim_final <- recits_faim[IsToKeep == "yes"]


# Ajout à ce lot les documents qui ont été classés à la main
recits_faim_classes_yes <- data_binded[IsToKeep == "yes", .(ID)]
recits_faim_classes_no <- data_binded[IsToKeep == "no", .(ID)]
setnames(recits_faim_classes_yes, old = "ID", new = "doc_id")

recits_faim_final <-
  rbind(recits_faim_final[, .(doc_id)], recits_faim_classes_yes[, .(doc_id)])

recits_faim_rejetes <- recits_faim[IsToKeep == "no"]

recits_faim_rejetes_final <-
  rbind(recits_faim_rejetes[, ID := doc_id][, .(ID)], recits_faim_classes_no[, .(ID)])


# Réintroduction de toutes les variables originales
recits_faim_raw <-
  fread('')

setnames(recits_faim_raw,
         old = c("U1", "ID"),
         new = c("text", "doc_id")) # Identifier les colonnes texte et doc_id

setcolorder(recits_faim_raw, c("doc_id", "text")) # Placer les deux colonnes importantes au début du tableau

recits_faim_final <-
  merge(recits_faim_final, recits_faim_raw, by = "doc_id")

fwrite(recits_faim_final,
       "")

recits_rejetes_final <-
  merge(recits_faim_rejetes_final,
        recits_faim_raw,
        by.x = "ID",
        by.y = "doc_id")
fwrite(recits_faim_rejetes_final,
       "")


# Validation manuelle de la classification supervisée

echantillon <- recits_faim_final[!doc_id %in% doc_idx$ID]
library(dplyr)
echantillon <- slice_sample(echantillon, n = 100)
echantillon <- echantillon[, .(doc_id, text)]

dir.create("data/verification")

for (i in 1:nrow(echantillon)) {
  write_file(
    paste(echantillon$doc_id[i], echantillon$text[i], sep = "\n"),
    paste0("data/verification/", echantillon$doc_id[i], ".txt")
  )
}


# Résultat: 94 vrais positifs et 6 faux positifs

# Retrait du corpus final des 6 faux positifs
faux_positifs <- c("1598207964",
                   "1598497327",
                   "1598207253",
                   "1600209818",
                   "1598783399",
                   "1598038487")
str(recits_faim_final)
recits_faim_final <- recits_faim_final[!doc_id %in% faux_positifs]
fwrite(recits_faim_final,
       "")


# Ajout aux documents rejetés les 6 faux positifs
recits_faim_rejetes_final <-
  append(recits_faim_rejetes_final$ID, faux_positifs)
recits_faim_rejetes_final <-
  recits_faim_raw[doc_id %in% recits_faim_rejetes_final]

fwrite(recits_faim_rejetes_final,
       "")

recits_faim_final <-
  fread("")
recits_faim_rejetes_final <-
  fread("")

fwrite(recits_faim_final[, .(doc_id)],
       "")
fwrite(
  recits_faim_rejetes_final[, .(doc_id)],
  ""
)

recits_faim_rejetes_final[!doc_id %in% doc_idx$ID, .N]
