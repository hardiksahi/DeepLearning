# Assignment 2
## Name: Hardik Sahi | WatID: 20743327

**Note: I have used gensim tokenizer to tokenize the data and  word count as features.**


The classification accuracy table for different scenarios is as follows:


Stopwords removed | Text features | Alpha |Accuracy(test set) 
--- | --- | --- | ---
yes | unigrams | 1.0 | 0.7197
yes | bigrams | 1.0 | 0.7222375
yes | unigrams+bigrams | 1.0 | 0.74605
no | unigrams | 0.9 | 0.72375
no | bigrams | 0.8 | 0.7517375
no | unigrams+bigrams | 0.5 | 0.7603


## Question1: Which condition performed better: with or without stopwords?
As it is clear from the table above, the **condition with stopwords performed better** for each of unigram, bigram and unigram+bigram cases.
A review is composed of stopwords between sentiment and the target object. The stopword removal can be problematic in the context of sentiment analysis because it might affect the context and change it into something that was not meant or worst, into something gibberish.


e.g. Original sentence: **I do not like icecream.**, Stopword removed: **like icecream**


Clearly stopword removal changes the very meaning of the sentence. 
We have to be very careful in choosing which stopwords to remove or not and that impacts the classifier performance.  

## Question2: Which condition performed better: unigrams, bigrams or unigrams+bigrams?
As can be seen from the table above, the performance of **unigram+bigram features is the best, followed by bigram and then unigram** for both with and without stopwords.
This is because the success of sentiment analysis classifier is heavily dependent on how well the context is captured. 
**Unigram does not capture any context** (surrounding words) and hence performs the worst.
**Bigram captures 1-neighborhood information** as context and hence performs better than unigram.
**Combining unigram and bigram captures much richer context** and hence gives the best accuracy on test set.
