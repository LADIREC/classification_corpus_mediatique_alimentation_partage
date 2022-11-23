###Select classified articles with randomforest and merge with initial corpus
##Author: Lisa Teichmann and Pascal Brisette, LaDiRec and McGill University
##Complete repo: https://github.com/LADIREC/ladirec_class_partage
##For more detailed information and results acquired through this method, please read our report "Document Classification with Machine Learning (Random Forest and SVM)"

##Load packages
#setwd()
library(data.table)
library(dplyr)

##Only select positively classified articles and add columns from corpus file
classed_up <- readRDS("results/20220301_LT_class_public_corpen_sub_rd_rf_upsample_alldocs_predicted.RDS")
table(classed_up$prediction)

classed_pos <- classed_up[classed_up$prediction=="1",]

corpus_all <- readRDS("data/20220224_LT_class_public_corpen_fullcorpus_tfidf.RDS")

names(corpus_all)[1] <- "doc_id"
corpus_all$doc_id <- as.character(corpus_all$doc_id)

joined_corpus <- left_join(classed_pos, corpus_all, 
                           by = c("doc_id"))

write.csv(joined_corpus, file="results/20221122_CRIEM_recitsfaim_supercat_20210927_corpen_rf_classified.csv")
