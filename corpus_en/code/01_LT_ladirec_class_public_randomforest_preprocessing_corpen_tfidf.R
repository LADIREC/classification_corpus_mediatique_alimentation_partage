###Classification with Random Forest: Pre-processing food-related articles and creating training and testing set
##Author: Lisa Teichmann, LaDiRec and McGill University
##Complete repo: https://github.com/LADIREC/ladirec_class_public
##!!!Only run this in case you have your own dataset. Due to license restrictions, we did not publish the dataset used in this script.
##!!!If you want to use our data please skip this script and proceed with 02_ladirec_class_public_randomforest_training_corpen

##Load packages
#setwd()

libs <- c("tidyverse","tm", "caret", "tokenizer",
          "readtext", "data.table", "text2vec", "lsa",
          "quanteda", "janitor", "dplyr", "tidyr")
lapply(libs, require, character.only = TRUE)


###Load classified data
#Dataset must contain a document ID column, a text column(pre-processed) and a tag column (with a binary classification), in our case this is "2rmv" and "2keep"
#About the training data: we classified a set of 1633 English articles into food-related and non-food-related based on classification guide (See: Guide de classifcation_V3)
#depending on the performance of the model we might need more training data
#setwd()
classified <- read.csv("20211018_LT_class_corpen_sub_rd_classified_2225.csv", sep="\t")

###Clean classified data

classified <- as.data.table(classified)
classified[, text := tolower(text)]

###Text cleaning (remove numericals, links, etc.)

classified$text <- classified$text %>%
  str_replace_all(., "^([mM]ontréal|[qQ]uébec|[pP]aris|[wW]ashington|[tT]oronto)\\s—", " ") %>%
  str_replace_all(., '[^[:alnum:]]', ' ') %>%
  str_replace_all(.,"\\b[:alpha:]{1,2}\\b", " ") %>%
  str_replace_all(., "[[:digit:]{5,}]", " ") %>%
  str_replace_all(., "\\b(www|https*|com|ca|mme|min|er|qc|bre|st)\\b", " ")

##Check the result
classified[1,text]

##Removing stopwords
lsastops <- list(mots = stopwords("english"))
setDT(lsastops)

stopwords_en <- tibble(word = lsastops[,mots])

classified <- classified %>% unnest_tokens(output = "word",
                                         input = "text",
                                         token = "words")

classified <- classified %>% anti_join(stopwords_en, by = "word")
classified <- classified %>% group_by(doc_id) %>% summarise(text=paste0(word, collapse = " "))

classified <- arrange(classified, doc_id)

##Make sure there is no whitespaces
classified$text <- stripWhitespace(classified$text)
classified$text <- str_trim(classified$text, side = "both")

classified[1]

##Look at the classification ratio of 2keep(food-related) and 2rmv(not food-related),ideally, this should be 50/50)
classified %>% count(class)

###Transforming the "class" column into factors with 0 and 1 values

classified = classified %>%
  mutate(class_num = recode(classified$class,"2keep"=1, "2rmv"=0)) %>% drop_na()

##delete NA's
classified %>% count(class_num)

##Randomforest takes factors as input
classified$class_num = as.factor(classified$class_num)

##Delete columns we won't need
classified$X<- NULL
classified$class=NULL

##Remove duplicates
classified <- classified[!duplicated(classified$doc_id),]

###Vectorize
tok_fun = word_tokenizer
it_train <- itoken(classified$text,
                   tokenizer = tok_fun,
                   ids = classified$doc_id,
                   progressbar = FALSE)
vocab = create_vocabulary(it_train)

###Create DTM
vectorizer <- vocab_vectorizer(vocab)
vocab_tfidf <- create_vocabulary(it_train)
vocab_tfidf <-  prune_vocabulary(vocab_tfidf,
                                 term_count_min = 20,
                                 doc_proportion_max = 0.5)
vectorizer_tfidf <- vocab_vectorizer(vocab_tfidf)
dtm_train_tfidf <- create_dtm(it_train, vectorizer_tfidf)
tfidf = TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train_tfidf, tfidf)

##Do the same for testset (optional)
# it_test <- tok_fun(test$text)
# it_test <- itoken(it_test, ids = test$doc_id,
#                   progressbar = FALSE)
# dtm_test_tfidf <- create_dtm(it_test, vectorizer_tfidf)
# dtm_test_tfidf <- transform(dtm_test_tfidf, tfidf)
# dim(dtm_train_tfidf)
# dim(dtm_test_tfidf)

dtm_train_tfidf_df <- as.data.frame.matrix(dtm_train_tfidf)
#dtm_test_tfidf_df <- as.data.frame.matrix(dtm_test_tfidf)
dtm_train_tfidf_df$IsToKeep <- classified$class_num
#dtm_test_tfidf_df$IsToKeep <- test$IsToKeep

###Load and preprocess complete corpus
corpus_en <- read_rds("20220223_CRIEM_recitsfaim_supercat_20210927_corpen_textonly_clean.RDS")
recits_faim <- corpus_en

##Delete all classified documents
recits_faim <- recits_faim[!duplicated(text)]
recits_faim <- recits_faim[!duplicated(doc_id)]

##Create DTM (document term matrix)
tok_fun = word_tokenizer
it_fin <- tok_fun(recits_faim$text)
it_fin <- itoken(it_fin, ids = recits_faim$doc_id,
                 progressbar = FALSE)

dtm_pruned_fin <- create_dtm(it_fin, vectorizer_tfidf)
dim(dtm_pruned_fin)
dtm_pruned_fin_df <- as.data.frame.matrix(dtm_pruned_fin)

##Create training and testing set
##Splitting the data into a training and testing set
trainIndex <- createDataPartition(dtm_train_tfidf_df$IsToKeep, p = 0.8,
                                  list = FALSE)

trainSet <- dtm_train_tfidf_df[trainIndex,]
testSet <- dtm_train_tfidf_df[-trainIndex,]

##The class column is called IsToKeep and needs to be a factor
##Check if set has IsToKeep column
#trainSet$IsToKeep
trainSet$IsToKeep <- as.factor(trainSet$IsToKeep)
testSet$IsToKeep <- as.factor(testSet$IsToKeep)

###Save for the next script
##this is the input data for the training script 02_ladirec_class_public_randomforest_training_corpen

saveRDS(dtm_pruned_fin_df, file="20220224_LT_class_public_corpen_fullcorpus_tfidf.RDS")
saveRDS(trainSet, file="20220224_LT_class_public_corpen_sub_rd_ml_trainset.RDS")
saveRDS(testSet, file="20220224_LT_class_public_corpen_sub_rd_ml_testset.RDS")


################################################
