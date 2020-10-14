# Assignment 3
## Name: Hardik Sahi | WatID: 20743327

**Command: python Assignment3.py train_pos_csv  train_neg_csv   val_pos_csv  val_neg_csv  test_pos_csv  test_neg_csv**

**Words most similar to "good":**
(word, similarity metric)
1. ('great', 0.7743538618087769)
2.  ('decent', 0.7490059733390808)
3. ('nice', 0.6821601986885071)
4. ('fantastic', 0.6788416504859924)
5. ('wonderful', 0.6200543642044067)
6. ('terrific', 0.617676854133606)
7. **('bad', 0.6100952625274658)**
8. ('excellent', 0.573219895362854)
9. ('amazing', 0.5627210140228271)
10. ('awesome', 0.5562898516654968)
11. ('reasonable', 0.5470976233482361)
12. ('impressive', 0.5454849600791931)
13. **('poor', 0.5417416095733643)**
14. ('superb', 0.5392881035804749)
15. **('terrible', 0.5245688557624817)**
16. ('perfect', 0.5152590274810791)
17. ('fabulous', 0.5150091648101807)
18. ('okay', 0.5025518536567688)
19. ('ok', 0.5003058910369873)
20. **('cheap', 0.48329734802246094)**


**Words most similar to "bad":**
(word, similarity metric)
1. **('good', 0.6100953221321106)**
2. ('horrible', 0.5857611298561096)
3. ('terrible', 0.5622707605361938)
4. ('awful', 0.5361939668655396)
5. ('poor', 0.5053960084915161)
6. ('weird', 0.46120935678482056)
7. ('funny', 0.4593249559402466)
8. ('disappointing', 0.45823749899864197)
9. ('strange', 0.4360940456390381)
10. ('stupid', 0.42548584938049316)
11. ('scary', 0.4192296266555786)
12. ('sad', 0.4164392054080963)
13. ('nasty', 0.41436767578125)
14. ('okay', 0.4133816361427307)
15. ('weak', 0.41187503933906555)
16. ('lousy', 0.406692773103714)
17. ('shabby', 0.40365397930145264)
18. ('loud', 0.395733118057251)
19. ('cheap', 0.394210547208786)
20. **('impressive', 0.3942086398601532)**

The classification accuracy table for different scenarios is as follows:


## Question: Are the words most similar to “good” positive, and words most similar to “bad” negative? Why this is or isn’t the case?

As it is clear from the lists of words similar to 'good' and 'bad' above, words **similar to 'good' are not all positive** and words **similar to 'bad' are not all negative**. 
This is because Word2Vec model encodes a word based on its context defined by the surrounding words. This means that the words which appear in similar context have similar word embedding. It is completely possible that the word 'bad' has appeared in similar context as word 'good' and hence the model provides them similar word embeddings.  


e.g. **It was a very good movie, It was a very bad movie**. In this example, words **good** and **bad** appear in similar context (defined by surrounding words). Hence word2Vec provides them similar embedding.
