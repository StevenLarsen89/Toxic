############## Loader Pakker #################

library(tidyverse)
library(magrittr)
library(qdap)
library(tm)
library(stringr)
library(text2vec)
library(tokenizers)
library(glmnet)
library(xgboost)
library(pROC)
library(caret)
library(parallel)
library(doParallel)


############## Loader Data #################

### TRÆNINGSSÆTTET ###

train <- read.csv("train.csv")
train <- select(train, c("id","comment_text", "toxic"))
    
    # Andelen af observationer, som er toxic
    sum(train$toxic)/nrow(train) # 0.09584448


# Da Træningssættet er meget stort, udtager jeg et subset, som skal bruges til at træne modellen
train_1 <- train %>% filter(toxic == 1) # Tager alle 'toxic' kommentarer med
train_0 <- train %>% filter(toxic == 0)
index <- sample(nrow(train_0), nrow(train_0)*0.25) # Tager kun 25% af de ikke 'toxic' kommentarer
train_0 <- train_0[index, ]
train <- rbind(train_1, train_0) # Samler træningssættet igen
train$set <- 1 # TRAIN = 1
    
    # Andelen af observationer, som er toxic
    sum(train$toxic)/nrow(train) # 0.297763


train$toxic <- as.factor(ifelse(train$toxic == 1, "Toxic","NonToxic"))
#Y <- train$toxic

rm(list=c("train_0", "train_1", "index"))

### TESTSÆTTET ###

test.nolabels <- read.csv("test.csv")
test.labels <- read.csv("test_labels.csv")
test <- cbind(test.nolabels, select(test.labels, toxic))
test$set <- 0 # TEST = 0
test <- test %>% filter(toxic != -1)

test$toxic <- as.factor(ifelse(test$toxic == 1, "Toxic","NonToxic")) 

data <- rbind(train, test)

rm(list=c("test.labels", "test.nolabels", "test", "train"))

############## Dataarbejde #################

data$comment_text <- replace_abbreviation(data$comment_text)  ## Fjerner forkortelser
data$comment_text <- replace_contraction(data$comment_text)   ## Fjerner kontraktioner
data$comment_text <- removeNumbers(data$comment_text)         ## Fjerner tal
data$comment_text <- removePunctuation(data$comment_text)     ## Fjerner tegn
data$comment_text <- tolower(data$comment_text)               ## Laver alle bogstaver smaa
data$comment_text <- rm_stopwords(data$comment_text, unlist = FALSE, separate = FALSE)
data$comment_text <- stripWhitespace(data$comment_text)       ## Fjerner whitespaces
data$comment_text <- data$comment_text %>% str_replace_all("[^a-zA-Z0-9]", " ") # fjerner alt der ikke er bogstaver eller tal

############## Sparse matrice test ##################

# normal_matrix <- matrix(0, nrow = 10000, ncol = 10000)
# sparse_matrix <- Matrix(0, nrow = 10000, ncol = 10000, sparse = TRUE)

# object.size(normal_matrix)
# 800000200 bytes
# object.size(sparse_matrix)
# 41632 bytes

# rm(list=c("m1", "m2"))


############## Text2vec #################

text_data <- data$comment_text %>% 
            itoken(tokenizer = word_tokenizer)

ordbog <- create_vocabulary(text_data, ngram = c(1L, 1L), stopwords = Top200Words) %>%  ### Laver ordbog ###
            prune_vocabulary(term_count_min = 3, 
                              doc_proportion_max = 0.3,
                              vocab_term_max = 3000) ### Pruner Ordbog ###

### Laver DocumentTermMatrix ###
vectorizer <- ordbog %>% vocab_vectorizer()
dtm <- create_dtm(text_data, vectorizer) 

### splitter datasættet igen ###
train.set <- data[, "set"] == 1
test.set <- data[, "set"] == 0
train.dtm <- dtm[train.set,]
test.dtm <- dtm[test.set,]

train_data <- data[train.set,]
test_data <- data[test.set,]

rm(list=c("dtm", "data", "test.set", "text_data", "train.set"))


############# CARET ###########

set.seed(133)
folds <- createFolds(train_data$toxic, k = 5)  # laver 5-fold CV


### DEFINERER CONTROLS ### 
controls <- trainControl(
  method = "cv",
  index = folds,
  number = 5,
  summaryFunction = twoClassSummary, 
  classProbs = TRUE,                  
  verboseIter = TRUE, 
  returnData = FALSE,
  allowParallel = FALSE
  )

### TRÆNER MODEL ###

### Multi core ###
# cluster <- makeCluster(detectCores() - 2)
# registerDoParallel(cluster)

set.seed(133)
m_rf <- train(
  y = Y,                                     
  x = train.dtm,                                          
  metric = "ROC",                                                
  method = "Rborist",                                           
  trControl = controls,
  nTree = 500,
  tuneGrid = expand.grid(predFixed = seq(10, 100, 10),
                         minNode = seq(1, 10, 1)
                        )       
)

set.seed(133)
m_xgb <- train(
  y = Y,                                     
  x = train.dtm,                                          
  metric = "ROC",                                                
  method = "xgbTree",                                           
  trControl = controls
)

# KØR EFTER TRÆNING #
# stopCluster(cluster)
# registerDoSEQ()


### Sammenligner modellern ###
max(m_rf$results$ROC) # max ROC-AUC 0.9125224
max(m_xgb$results$ROC) # max ROC-AUC 0.934225

print(m_xgb$bestTune)
# nrounds    max_depth    eta     gamma     colsample_bytree    min_child_weight    subsample
# 150        3            0.4     0         0.8                 1                   0.75

print(m_rf$bestTune)
# predFixed   minNode
# 10          6

# predFixed number of trial predictors for a split (mtry).
# minNode oberservation i hvert blad

### Prædiktere på testsættet ###
Y_pred_rf <- predict(m_rf, newdata = test.dtm, type = "prob")
Y_pred <- predict(m_xgb, newdata = test.dtm, type = "prob")


# BEREGNER ROC-KURVE
ROC_test_xgb <- roc(test_data$toxic, Y_pred$Toxic)
ROC_test_rf <- roc(test_data$toxic, Y_pred_rf$Toxic)

# BEREGNER AUC
auc(ROC_test_xgb) # 0.9352
auc(ROC_test_rf)  # 0.918

# PLOTTER ROC-KURVER
ggroc(ROC_test_xgb, legacy.axes = TRUE) +
  xlab("Falsk positiv rate") + 
  ylab("Sand positiv rate") + 
  ggtitle("ROC: Gradient Boosted Trees") +
  theme_bw()


# BEREGNER IMPORTANCE AF ORD
importance <- varImp(m_xgb)$importance

top20 <- head(importance, n = 20)
top20$ord <- row.names(top20)

ggplot(top20, aes(x = reorder(ord, Overall), y = Overall)) +
  theme_bw() +
  geom_col(aes(fill = reorder(ord, Overall))) +
  coord_flip() +
  xlab(" ") + 
  ylab("Importance") + 
  theme(legend.position="none")



