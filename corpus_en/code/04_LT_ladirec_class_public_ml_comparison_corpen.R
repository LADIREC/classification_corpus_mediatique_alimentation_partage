###Train machine learning models and compare to identify the model with the best results of Support-vector models (SVM), k-nearest neighbours (KNN), Latent Dirichlet allocation (LDA), and Classification And Regression Trees (CART)
##Author: Lisa Teichmann and Pascal Brisette, LaDiRec and McGill University
##Complete repo: https://github.com/LADIREC/ladirec_class_public
##For more detailed information and results acquired through this method, please read our report "Document Classification with Machine Learning (Random Forest and SVM)"

##Load packages
#setwd()
library(caret)
library(mlbench)
library(e1071)

##Load preprocessed data from 01_LT_ladirec_class_public_randomforest_preprocessing_corpen.R
##Files are in the "data" folder

dtm_pruned_fin_df <- readRDS("20220224_LT_class_public_corpen_fullcorpus_tfidf.RDS")
trainSet <- readRDS("20220224_LT_class_public_corpen_sub_rd_ml_trainset.RDS")
testSet<- readRDS("20220224_LT_class_public_corpen_sub_rd_ml_testset.RDS")

trainSet <- trainSet %>%
  column_to_rownames('X')
testSet <- testSet %>%
  column_to_rownames('X')
dtm_pruned_fin_df <- dtm_pruned_fin_df %>%
  column_to_rownames('X')

trainSet$IsToKeep <- as.factor(trainSet$IsToKeep)
testSet$IsToKeep <- as.factor(testSet$IsToKeep)

###Train classifiers without feature selection (very time-intensive)
## Define common parameters
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

## CART
fit.cart <- train(IsToKeep ~ ., data = trainSet, method = "rpart", trControl = control)
#saveRDS(fit.cart, file="20220308_LT_class_public_corpen_ml_carttrain.RDS")

## LDA
#fit.lda <- train(IsToKeep ~ ., data = trainSet, method = "lda", trControl = control)
#saveRDS(fit.lda, file="20220308_LT_class_public_corpen_ml_ldatrain.RDS")

## SVM
fit.svm <- train(IsToKeep ~ ., data = trainSet, method = "svmRadial", trControl = control)
#saveRDS(fit.svm, file="20220308_LT_class_public_corpen_ml_svmtrain.RDS")

## KNN
fit.knn <- train(IsToKeep ~ ., data = trainSet, method = "knn", trControl = control)
#saveRDS(fit.knn, file="20220308_LT_class_public_corpen_ml_knntrain.RDS")

## Random Forest
#fit.rf <- train(IsToKeep ~ ., data = trainSet, method = "rf", trControl = control, verbose = TRUE)
#saveRDS(fit.rf, file="20220308_LT_class_public_corpen_ml_rftrain.RDS")

results <- resamples(list(CART = fit.cart, SVM = fit.svm, KNN = fit.knn))

### Compile results
comparaison_modeles <- summary(results)

saveRDS(comparaison_modeles, '2022-3-9_ComparedModels_corpen.RDS')

scales <- list(x = list(relation = "free"), y=list(relation="free"))
bwplot(results, scales = scales)

scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|")

##dot plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)

##xyplot plots to compare models
xyplot(results, models=c("RF", "SVM"))


###Train dataset with feature selection with mean decrease Gini (dtm_train_pruned_imp_df)

###Feature selection with Gini
importance <- randomForest::importance(classifier_gini, type = 2)

my_df <- data.table(variable = attributes(importance)$dimnames[[1]],
                    valeur = importance)

variables2keep <- my_df[my_df$valeur.MeanDecreaseGini>0.5, 1]

##Convert the original dataset to a data.table format
dtm_train_pruned_df <- as.data.frame.matrix(trainSet)
dtm_test_pruned_df <- as.data.frame.matrix(testSet)

##Subset the training and testing set by variables deemed important according to thier Gini coefficient
dtm_train_pruned_imp_df <- dtm_train_pruned_df[,colnames(dtm_train_pruned_df) %in% variables2keep$variable]
dtm_test_pruned_imp_df <- dtm_test_pruned_df[,colnames(dtm_test_pruned_df) %in% variables2keep$variable]

dtm_train_pruned_imp_df$IsToKeep <- trainSet$IsToKeep
dtm_test_pruned_imp_df$IsToKeep <- testSet$IsToKeep

###Train models and compare

##Define common parameters
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(7)

# CART
fit.cart.fs <- train(IsToKeep ~ ., data = dtm_train_pruned_imp_df, method = "rpart", trControl = control)

# LDA
fit.lda.fs <- train(IsToKeep ~ ., data = dtm_train_pruned_imp_df, method = "lda", trControl = control)

# SVM
fit.svm.fs <- train(IsToKeep ~ ., data = dtm_train_pruned_imp_df, method = "svmRadial", trControl = control)

# KNN
fit.knn.fs <- train(IsToKeep ~ ., data = dtm_train_pruned_imp_df, method = "knn", trControl = control)

# Random Forest
fit.rf.fs <- train(IsToKeep ~ ., data = dtm_train_pruned_imp_df, method = "rf", trControl = control)

results.fs <- resamples(list(CART = fit.cart.fs, LDA = fit.lda.fs, SVM = fit.svm.fs, KNN = fit.knn.fs, RF = fit.rf.fs))

# Synthétiser les résultats
comparaison_modeles_fs <- summary(results.fs)

write_rds(comparaison_modeles_fs, 'ComparedModels_featureselection_corpen.RDS')

scales.fs <- list(x = list(relation = "free"), y=list(relation="free"))
bwplot(results.fs, scales = scales.fs)

scales.fs <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales.fs, pch = "|")

##dot plots of accuracy
scales.fs <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results.fs, scales=scales.fs)

##xyplot plots to compare models
xyplot(results.fs, models=c("RF", "SVM"))
