###Classification with Random Forest: Classifier training and prediction
##Author: Lisa Teichmann and Pascal Brisette, LaDiRec and McGill University
##Complete repo: https://github.com/LADIREC/ladirec_class_public
##For more detailed information and results acquired through this method, please read our report "Document Classification with Machine Learning (Random Forest and SVM)"

##Load packages
#setwd()

library(randomForest)
library(caret)
library(lsa)
library(caret)
library(data.table)
library(dplyr)
library(tibble)

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


#####################Train the Random Forest classifier########################################

##Fine-tune the classifier by finding the best mtry (lowest OOB error rate)
##Depending on the GPU this can take up to a couple of hours

mtry <- tuneRF(x=trainSet[,-ncol(trainSet)],
               y = trainSet[,ncol(trainSet)],
               mtryStart=100,
               ntreeTry = 600,
               stepFactor=1.5,
               improve = 0.01,
               trace = TRUE,
               plot = TRUE)

best.m <- mtry[mtry[,2] == min(mtry[,2]), 1]
print(mtry)
print(best.m)

##In case it is too slow, we found the best mtry was 52
#best.m <- 225

##Train the random forest model
classifier <- randomForest(x = trainSet[, -ncol(trainSet)],
                           y = trainSet[, ncol(trainSet)],
                           mtry = best.m,
                           ntree = 600,
                           importance = TRUE,
                           proximity = TRUE)

##Predict the class of the test data
rf_pred <- predict(classifier, newdata = testSet[,-ncol(testSet)])
cm <- table(testSet[, ncol(testSet)], rf_pred)

##Look at the results of the confusion matrix, especially accuracy, sensitivity, and specificity
print(confusionMatrix(cm))

##Look at accuracy for each document
#print(rf_pred)


##Cross-validate by reading the misclassified articles
##!!!Cross-validation can only be done if the complete text of each article is preserved
##!!!Please skip the next lines if you use our dataset

##Look at the misclassified articles (false positives and false negatives)
# rf_pred_df <- as.data.frame(rf_pred)
# rf_pred_df <- tibble::rownames_to_column(rf_pred_df, "doc_id")
# 
# crossval <- merge(rf_pred_df, classified[, c("doc_id", "class_num")], by="doc_id")
# ##crossval_false <- which(crossval$rf_pred !=crossval$class_num)
# ##crossval$istrue <- ifelse(crossval$rf_pred == crossval$class_num, TRUE, FALSE)
# crossval_false <- crossval[crossval$rf_pred !=crossval$class_num, ]
# 
# ##add text column
# crossval_false <- merge(crossval_false, classified[, c("doc_id", "text")], by="doc_id")

##write.csv(crossval_false, file="20221224_LT_class_corpen_sub_rs_randomforest_misclass_corpen.csv")

#####################Feature Selection#############################################
##We tested various options for feature selection: Gini coefficient, upsampling, and Boruta

###Option 1: Feature selection with mean-decrease Gini-coefficient

##We can use the model we have already trained
importance <- randomForest::importance(classifier, type = 2)

classifier_imp <- data.table(variable = attributes(importance)$dimnames[[1]],
                    value = importance)

##Set a threshold for the Gini coefficient
variables2keep <- classifier_imp[classifier_imp$value.MeanDecreaseGini>0.5, 1]

##Convert the original dataset to a data.table format
dtm_train_pruned_df <- as.data.frame.matrix(trainSet)
dtm_test_pruned_df <- as.data.frame.matrix(testSet)

##Subset the training and testing set by variables deemed important according to thier Gini coefficient
dtm_train_pruned_imp_df <- dtm_train_pruned_df[,colnames(dtm_train_pruned_df) %in% variables2keep$variable]
dtm_test_pruned_imp_df <- dtm_test_pruned_df[,colnames(dtm_test_pruned_df) %in% variables2keep$variable]

##Append the class column
dtm_train_pruned_imp_df$IsToKeep <- trainSet$IsToKeep
dtm_test_pruned_imp_df$IsToKeep <- testSet$IsToKeep

##Finetune (mtry will differ with fewer features)
gini_mtry <- tuneRF(x=dtm_train_pruned_imp_df[,-ncol(dtm_train_pruned_imp_df)],
               y = dtm_train_pruned_imp_df[,ncol(dtm_train_pruned_imp_df)],
               mtryStart=100,
               ntreeTry = 600,
               stepFactor=1.5,
               improve = 0.01,
               trace = TRUE,
               plot = TRUE)

gini_best.m <- gini_mtry[gini_mtry[,2] == min(gini_mtry[,2]), 1]
print(gini_mtry)
print(gini_best.m)

##In case it is too slow, we found the best mtry was 52
#gini_best.m <- 67


##Train the model with selected features from Gini coefficient
classifier_gini <- randomForest(x = dtm_train_pruned_imp_df[, -ncol(dtm_train_pruned_imp_df)],
                           y = dtm_train_pruned_imp_df[, ncol(dtm_train_pruned_imp_df)],
                           mtry = gini_best.m,
                           ntree = 600,
                           importance = TRUE,
                           proximity = TRUE)

##Predict the class of the test data
rf_pred_gini <- predict(classifier_gini, newdata = dtm_test_pruned_imp_df[,-ncol(dtm_test_pruned_imp_df)])
cm_gini <- table(dtm_test_pruned_imp_df[, ncol(dtm_test_pruned_imp_df)], rf_pred_gini)

##Look at the results of the confusion matrix, especially accuracy, sensitivity, and specificity
print(confusionMatrix(cm_gini))

###Option 2a: Upsample by increasing the number of 2rmv (0-value)
##For some reason the upsample function from the groupdata2 package does not work on the full trainSet which is why we wrote our own script
##The number of variables is too large for any of the functions

# library(groupdata2)
# upsample <- upsample(trainSet, cat_col = "IsToKeep")
# 
# table(upsample$IsToKeep)
# 
# ##Train the model with the upsampled trainSet
# classifier_up <- randomForest(x = upsample[, -ncol(upsample)],
#                                   y = upsample[, ncol(upsample)],
#                                   mtry = 19,
#                                   ntree = 600,
#                                   importance = TRUE,
#                                   proximity = TRUE)
# 
# rf_pred_up <- predict(classifier_up, newdata = testSet[,-ncol(testSet)])
# cm_up <- table(testSet[, ncol(testSet)], classifier_up)
# 
# print(confusionMatrix(cm_up))

##Workaround for upsampling
trainSet_up <- trainSet %>%
        rownames_to_column('doc_id')

###it is important to conserve the doc_id by converting it to a column before splitting

docs2keep <- trainSet %>%
        rownames_to_column('doc_id') %>%
        group_by(IsToKeep) %>%
        filter(IsToKeep == 1) %>%
        column_to_rownames('doc_id')

docs2rmv <- trainSet %>%
        rownames_to_column('doc_id') %>%
        group_by(IsToKeep) %>%
        filter(IsToKeep == 0) %>%
        column_to_rownames('doc_id')

##increase 2rmv by the number of 2keep by cloning them to get a balanced set
docs2rmv_clone <- docs2rmv
docs2rmv <- bind_rows(docs2rmv, docs2rmv_clone)
docs2rmv<-docs2rmv[sample(1:nrow(docs2rmv),(nrow(docs2keep)), replace=FALSE),]
        
trainSet_up <- bind_rows(docs2keep,docs2rmv)

##See if training set is balanced
table(trainSet_up$IsToKeep)

##remove rows with NA values
trainSet_up <- trainSet_up[complete.cases(trainSet_up),]

##Balanced?
table(trainSet_up$IsToKeep)

##fine-tune
mtry_up <- tuneRF(x = trainSet_up[, -ncol(trainSet_up)],
                  y = trainSet_up[, ncol(trainSet_up)], 
                  stepFactor=1.5, improve=1e-5, ntree=500)

best_mtry_up <- mtry_up[mtry_up[,2] == min(mtry_up[,2]), 1]
print(mtry_up)
print(best_mtry_up)

##In case it is too slow, we found the best mtry was 52
#best_mtry_up <- 79

###Train the model
classifier_up <- randomForest(x = trainSet_up[, -ncol(trainSet_up)],
                              y = trainSet_up[, ncol(trainSet_up)],
                              mtry = best_mtry_up,
                              ntree = 600,
                              importance = TRUE,
                              proximity = TRUE)

rf_pred_up <- predict(classifier_up, newdata = testSet[,-ncol(testSet)])
cm_up <- table(testSet[, ncol(testSet)], rf_pred_up)

print(confusionMatrix(cm_up))

###Option 2b: Upsample and Gini
library(groupdata2)
upsample_gini <- upsample(dtm_train_pruned_imp_df, cat_col = "IsToKeep")
table(upsample_gini$IsToKeep)

##Finetune

mtry_gini_up <- tuneRF(x=upsample_gini[,-ncol(upsample_gini)],
               y = upsample_gini[,ncol(upsample_gini)],
               mtryStart=100,
               ntreeTry = 600,
               stepFactor=1.5,
               improve = 0.01,
               trace = TRUE,
               plot = TRUE)

best.m_gini_up <- mtry_gini_up[mtry_gini_up[,2] == min(mtry_gini_up[,2]), 1]
print(mtry_gini_up)
print(best.m_gini_up)

##In case it is too slow, we found the best mtry was 52
#best_mtry_up <- 100

##Train the model
classifier_upgini <- randomForest(x = upsample_gini[, -ncol(upsample_gini)],
                                  y = upsample_gini[, ncol(upsample_gini)],
                                  mtry = best.m_gini_up,
                                  ntree = 600,
                                  importance = TRUE,
                                  proximity = TRUE)

rf_pred_upgini <- predict(classifier_upgini, newdata = testSet[,-ncol(testSet)])
cm_upgini <- table(testSet[, ncol(testSet)], rf_pred_upgini)

print(confusionMatrix(cm_upgini))


###Option 3: Feature selection with Boruta
#install.packages("Boruta")
library(Boruta)
set.seed(123)

##Input is the IsToKeep (2keep/2rmv) variable in the "data_df" dataframe

##Renaming "shadow" variable (attributes with names starting from "shadow" are reserved for internal use.) 
boruta.train <- trainSet
boruta.train$shadow <- NULL

##Train the model
boruta.train <- Boruta(IsToKeep ~ ., boruta.train, doTrace = 2)
print(boruta.train)

##plot
# plot(boruta.train, xlab = "", xaxt = "n")
# lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
#          boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
#  names(lz) <- colnames(boruta.train$ImpHistory)
#  Labels <- sort(sapply(lz,median))
#  axis(side = 1,las=2,labels = names(Labels),
#         at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

#get values for tentative attributes (Tentative attributes have importance so close to their best shadow attributes that Boruta is not able to make a decision with the desired confidence in default number of random forest runs.)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

##final plot
# plot(final.boruta, xlab = "", xaxt = "n")
# lz<-lapply(1:ncol(final.boruta$ImpHistory),function(i)
#         final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
# names(lz) <- colnames(final.boruta$ImpHistory)
# Labels <- sort(sapply(lz,median))
# axis(side = 1,las=2,labels = names(Labels),
#      at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7)

#obtain the list of confirmed attributes
boruta_confattr <- getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)

print(boruta.df)

#saveRDS(boruta_confattr, file="20220301_LT_class_public_corpen_sub_rd_boruta_confirmed_attributes.RDS")

###Subset train and testset by selected features from Boruta
##Set a threshold for the Gini coefficient
boruta_variables2keep <- boruta_confattr

##Convert the original dataset to a data.table format
boruta_dtm_train_pruned_df <- as.data.frame.matrix(trainSet)
boruta_dtm_test_pruned_df <- as.data.frame.matrix(testSet)

##Subset the training and testing set by variables deemed important according to thier Gini coefficient
boruta_dtm_train_pruned_df <- boruta_dtm_train_pruned_df[,colnames(boruta_dtm_train_pruned_df) %in% boruta_variables2keep]
boruta_dtm_test_pruned_df <- boruta_dtm_test_pruned_df[,colnames(boruta_dtm_test_pruned_df) %in% boruta_variables2keep]

##Append the class column
boruta_dtm_train_pruned_df$IsToKeep <- trainSet$IsToKeep
boruta_dtm_test_pruned_df$IsToKeep <- testSet$IsToKeep


# saveRDS(boruta_dtm_train_pruned_df, file="20220301_LT_class_public_corpen_sub_rd_boruta_trainset.RDS")
# saveRDS(boruta_dtm_test_pruned_df, file="20220301_LT_class_public_corpen_sub_rd_boruta_testset.RDS")


###Train randomforest with selected features by Boruta

##find best mtry

mtry_boruta <- tuneRF(x=boruta_dtm_train_pruned_df[,-ncol(boruta_dtm_train_pruned_df)],
               y = boruta_dtm_train_pruned_df[,ncol(boruta_dtm_train_pruned_df)],
               mtryStart=50,
               ntreeTry = 600,
               stepFactor=1.5,
               improve = 0.01,
               trace = TRUE,
               plot = TRUE)

best.m_boruta <- mtry_boruta[mtry_boruta[,2] == min(mtry_boruta[,2]), 1]
print(mtry_boruta)
print(best.m_boruta)

##We found the best mtry for boruta to be 10
#best.m_boruta <- 23

#run the model
set.seed(333)
classifier_boruta <- randomForest(x = boruta_dtm_train_pruned_df[, -ncol(boruta_dtm_train_pruned_df)],
                           y = boruta_dtm_train_pruned_df[, ncol(boruta_dtm_train_pruned_df)],
                           mtry = best.m_boruta,
                           ntree = 600,
                           importance = TRUE,
                           proximity = TRUE,)

rf_pred_boruta <- predict(classifier_boruta, newdata = boruta_dtm_test_pruned_df[,-ncol(boruta_dtm_test_pruned_df)])
cm_boruta <- table(boruta_dtm_test_pruned_df[, ncol(boruta_dtm_test_pruned_df)], rf_pred_boruta)

print(confusionMatrix(cm_boruta))

###Look at the misclassified articles (false positives and false negatives)
###!!!Only applicable if full text is available in "text" and dataframe includes a "doc_id" column

# rf_pred_df <- as.data.frame(rf_pred_boruta)
# testSet_fs_docid <- tibble::rownames_to_column(testSet_fs, "doc_id")
# rf_pred_df <- tibble::rownames_to_column(rf_pred_df, "doc_id")

##compare to original class
#crossval <- merge(rf_pred_df, data_df[, c("doc_id", "class_num")], by="doc_id")
#crossval_false <- which(crossval$rf_pred !=crossval$class_num)
#crossval$istrue <- ifelse(crossval$rf_pred == crossval$class_num, TRUE, FALSE)
#crossval_false <- crossval[crossval$rf_pred_fs !=crossval$class_num, ]

##add text column
#crossval_false <- merge(crossval_false, data_df[, c("doc_id", "text")], by="doc_id")

#write.csv(crossval_false, file="20210825_LT_class_corpen_sub_rs_randomforext_1633class_false.csv")

###Additional: Balance dataset of selected features from Boruta by upsampling (increasing 2rmv)
up_train_boruta <- upSample(x = boruta_dtm_train_pruned_df[, -ncol(boruta_dtm_train_pruned_df)],
                     y = boruta_dtm_train_pruned_df$IsToKeep)                         
table(up_train_boruta$Class)

##Finetune
mtry_boruta_up <- tuneRF(x=up_train_boruta[,-ncol(up_train_boruta)],
                      y = up_train_boruta[,ncol(up_train_boruta)],
                      mtryStart=50,
                      ntreeTry = 600,
                      stepFactor=1.5,
                      improve = 0.01,
                      trace = TRUE,
                      plot = TRUE)

best.m_boruta_up <- mtry_boruta_up[mtry_boruta_up[,2] == min(mtry_boruta_up[,2]), 1]
print(mtry_boruta_up)
print(best.m_boruta_up)


##Train
classifier_up_boruta <- randomForest(x = up_train_boruta[, -ncol(up_train_boruta)],
                           y = up_train_boruta[, ncol(up_train_boruta)],
                           mtry = best.m_boruta_up,
                           ntree = 600,
                           importance = TRUE,
                           proximity = TRUE)

rf_up_boruta <- predict(classifier_up_boruta, newdata = boruta_dtm_test_pruned_df[,-ncol(boruta_dtm_test_pruned_df)])

cm_up_boruta <- table(boruta_dtm_test_pruned_df[, ncol(boruta_dtm_test_pruned_df)], rf_up_boruta)

print(confusionMatrix(cm_up_boruta))


#######################################################################


