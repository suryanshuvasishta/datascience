#Topic Modeling and Text Mining on Annual Employee Satisfaction Survey
#
#Developed by suryanshuvasishta
#additions and omissions made the original code for privacy and anonymity
#
#libraries----------------------------------
library(tm)
library(qdap)
library(wordcloud)
library(ggplot2)
library(RWeka)
library(RWekajars)
library(stringr)
library(topicmodels)
library(tidytext)
library(dplyr)
#inputs--------------------------------------
#import data as csv. The flat file in question should have one comment record per line with headers as (column1) doc_id (column2) text 
#example format has been uploaded to git for more clarity

#data import---------------------------------
q <- read.csv("C:/Users/612024760/OneDrive - BT Plc/_gba/ys2019-2/gbs/q1.csv", stringsAsFactors = FALSE)
str(q)
nrow(q)

#removing unnecessary characters-------------
q = q[-which(is.na(q$text)),]
nrow(q)

#creating Data frame source and corpora------
q_corpus <- VCorpus(DataframeSource(q))
#random check on corpus content
content(q_corpus[[2000]])

#preprossing data using tm_map---------------
q_corpus <- tm_map(q_corpus, removePunctuation) #removing punctuation
q_corpus <- tm_map(q_corpus, content_transformer(tolower)) #transforming to lowercase
length(q_corpus) #no loss in number of comments detected
q_corpus <- tm_map(q_corpus, removeWords, stopwords("en")) #stop words removal
#incase of manual stopwords removal
#stop <- c(stopwords(kind = "en"), c("work", "working")) 
#q1_corpus <- tm_map(q1_corpus, removeWords, stop)
q_corpus <- tm_map(q_corpus, stripWhitespace) #whitespace removal
#q1_corpus <- tm_map(q1_corpus, stemDocument) #word stemming if needed
length(q_corpus) #no loss in comments detected

#Defining a new data frame with cleaned corpus and unique ID----
qdf <- data.frame(text = sapply(q_corpus, as.character), stringsAsFactors = F)
qdf$ID<- as.character(seq.int(nrow(qdf)))
nrow(qdf) #same as q1 corpus

#Creating term document matrix and document term matrix----
tdm_q <- TermDocumentMatrix(q_corpus)
dtm_q <- DocumentTermMatrix(q_corpus)
#Creating tdm --> matrix
tdm_m_q <- as.matrix(tdm_q) 
dim(tdm_m_q) 
tdm_m_q[100:104, 100:150]
#Adding row sums to tdm and sorting in decreasing order 
tdm_m_q_tf <- rowSums(tdm_m_q)
tdm_m_q_tf <- sort(tdm_m_q_tf, decreasing = TRUE)

#quick and dirty analysis using qdap package-----
#Trying qdap:: qdap will automatically pre-process data, stopwords are defined as top 200
freq1 <- freq_terms(q$text, top = 20, at.least = 5, stopwords = "Top200Words")
plot(freq1)
#qdap wordcloud
wordcloud(freq1$WORD, freq1$FREQ, colors = "blue")

#detailed analysis::word frequencies---------
#word frequency plot
par(mar = c(7,4,4,4))
barplot(tdm_m_q_tf[1:20], col = "purple", las = 2)
#word cloud
word_freq <- data.frame(term = names(tdm_m_q_tf), num = tdm_m_q_tf)
wordcloud(word_freq$term, word_freq$num, max.words = 100, colors = "blue")
#detailed analysis::word cluster analysis----
#dendrogram clusters
dist1 <- dist(word_freq$num[1:30])
hc1 <- hclust(dist1)
plot(hc1, word_freq$term[1:30])
#Checking another variation of the dendrogram with reduced sparcity
tdm_q_rs <- removeSparseTerms(tdm_q, sparse = 0.975)
dim(tdm_q_rs)
tdm_q_rs_m <- as.matrix(tdm_q_rs)
tdm_q_rs_m_dist <- dist(tdm_q_rs_m)
tdm_q_rs_m_dist_hc <- hclust(tdm_q_rs_m_dist)
plot(tdm_q_rs_m_dist_hc)
#detailed analysis::bigram and trigram analysis------
#Bigram analysis
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}
tdm_q_bigram <- DocumentTermMatrix(q_corpus, control = list(tokenize = tokenizer))
tdm_q_bigram
tdm_q_bigram_rs <- removeSparseTerms(tdm_q_bigram, sparse = 0.999)
tdm_q_bigram_m <- as.matrix(tdm_q_bigram_rs)
tdm_q_bigram_tf <- colSums(as.matrix(tdm_q_bigram_m))
tdm_q_bigram_tf <- sort(tdm_q_bigram_tf, decreasing = TRUE)
bi_words <- names(tdm_q_bigram_tf)
#str_subset(bi_words1, "^work")
wordcloud(bi_words, tdm_q_bigram_tf, max.words = 100, colors = "blue")
par(mar = c(11,3,3,5))
barplot(tdm_q_bigram_tf[1:50], col = "purple", las = 2)
#Plotting a bigram dendrogram
tdm_q_bigram_tf <- sort(tdm_q_bigram_tf, decreasing = TRUE)
word_freq1bi <- data.frame(term = names(tdm_q_bigram_tf), num = tdm_q_bigram_tf)
dist_bi <- dist(word_freq1bi$num[1:30])
hc1bi <- hclust(dist_bi)
plot(hc1bi, word_freq1bi$term[1:30])
#Trigram analysis
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 3, max = 3))
}
tdm_q_tri <- DocumentTermMatrix(q_corpus, control = list(tokenize = tokenizer))
tdm_q_tri
tdm_q_tri_rs <- removeSparseTerms(tdm_q_tri, sparse = 0.999)
tdm_q_tri_m <- as.matrix(tdm_q_tri_rs)
tdm_q_tri_tf <- colSums(as.matrix(tdm_q_tri_m))
tdm_q_tri_tf <- sort(tdm_q_tri_tf, decreasing = TRUE)
tri_words1 <- names(tdm_q_tri_tf)
#str_subset(tri_words1, "^work")
wordcloud(tri_words1, tdm_q_tri_tf, max.words = 100, colors = "blue")
par(mar = c(11,3,3,5))
barplot(tdm_q_tri_tf[1:50], col = "purple", las = 2)
#Plotting a trigram dendrogram
tdm_q_tri_tf <- sort(tdm_q_tri_tf, decreasing = TRUE)
word_freq1tri <- data.frame(term = names(tdm_q_tri_tf), num = tdm_q_tri_tf)
dist_tri <- dist(word_freq1tri$num[1:30])
hc1tri <- hclust(dist_tri)
plot(hc1tri, word_freq1tri$term[1:30])

#topic modelling using LDA-----
#Try variations with 2, 4, 6 and 8 clusters
dtm_q<-dtm_q[-which(apply(dtm_q,1,sum)==0),] #removing from dtm where sum is 0
lda1 <- LDA(dtm_q, k = 8, control = list(seed = 1234))
str(lda1)
lda1topics <- tidy(lda1, matrix='beta')
lda1topics
View(lda1topics)
top_terms <- lda1topics %>% group_by(topic) %>% top_n(15, beta) %>% ungroup() %>% arrange(topic, -beta)
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
lda1gamma <- tidy(lda1, matrix='gamma')
lda1gamma
lda1gammaclassification <- lda1gamma %>% group_by(document) %>% top_n(1, gamma) %>% ungroup()
table(lda1gammaclassification$topic)
nrow(lda1gammaclassification)
str(lda1gammaclassification)

qfinal <- left_join(qdf, lda1gammaclassification, by = c('ID' = 'document'))
nrow(qfinal)
write.csv(qfinal, "output.csv", row.names = F)
# write.csv(lda1gammaclassification, "lda1gammaclassification.csv", row.names = F)
