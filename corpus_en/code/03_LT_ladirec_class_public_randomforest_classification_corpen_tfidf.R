###Classification with Random Forest: Classify all unclassified documents of the corpus
##Author: Lisa Teichmann and Pascal Brisette, LaDiRec and McGill University
##Complete repo: https://github.com/LADIREC/ladirec_class_public
##For more detailed information and results acquired through this method, please read our report "Document Classification with Machine Learning (Random Forest and SVM)"

##Load packages
library("tm")
library(dplyr)
library(tidyr)
library(caret)
library(randomForest)
library(ggplot2)
#setwd()

libs <- c("tidyverse","tm", "caret", "tokenizer",
          "readtext", "data.table", "text2vec", "lsa",
          "quanteda", "janitor")
lapply(libs, require, character.only = TRUE)

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

##Classification
predict_final <- predict(classifier_up, newdata = dtm_pruned_fin_df,
                         type = "class")

##Create a dataframe with predictions and doc_id's
recits_faim_prediction <- tibble(doc_id = names(predict_final),
                                 prediction = predict_final)

##Visualize the distribution
table(recits_faim_prediction$prediction)
ggplot(data = recits_faim_prediction, aes(x = predict_final)) +
  geom_bar()

#saveRDS(recits_faim_prediction, file= "20220301_LT_class_public_corpen_sub_rd_rf_upsample_alldocs_predicted.RDS")

###Read a subset of articles to manually evaluate the efficacy
###!!!!!Only run this if you have your own dataset with full text

# recits_faim_prediction_df <- arrange(recits_faim_prediction, doc_id)
# 
# recits_faim_mots_composes <- readRDS('20220223_CRIEM_recitsfaim_supercat_20210927_corpen_textonly_clean.RDS') %>% select(doc_id, text)
# 
# recits_faim_prediction_df <- left_join(recits_faim_prediction_df, recits_faim_mots_composes, by = "doc_id")
# 
# ##Extract 100 articles to read and create txt files for each (for ingestion to Recogito)
# recits_sample <- sample_n(recits_faim_prediction_df, size = 100)
# 
# write.csv(recits_sample, "20220302_LT_corpen_randomforest_classified_random100.csv")

##write separate textfiles for each artcile (for annotation in Recogito)
#for (i in 1:nrow(recits_sample)) {
#   write_file(paste(recits_sample$doc_id[i], 
#                    recits_sample$prediction[i],
#                    recits_sample$text[i], sep = "\n"),
#              paste0("data/text2class/", recits_sample$doc_id[i], ".txt"))
# }
##After manual classification, we can check which articles are misclassified according to the model
# crossval <- read.csv("20220305_YC_corpen_randomforest_classified_random100._crossval.csv")
# crossval = crossval %>%
#   mutate(verification = recode(crossval$verification,"2keep"=1, "2rmv"=0)) %>% drop_na()
# 
# crossval$istrue <- ifelse(crossval$prediction == crossval$verification, TRUE, FALSE)
# crossval_false <- crossval[crossval$prediction !=crossval$verification, ]
# 
# write.csv(crossval_false, "20220310_LT_corpen_randomforest_classified_random100_crossval_false.csv")


