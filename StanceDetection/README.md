# StanceDetection
Data can be found at : https://github.com/FakeNewsChallenge/fnc-1

Training DataSet + Validation/ Dev DataSet : 39843 examples + 10129 examples <br />
Test set: 25413 examples <br />
Competition Dataset: 25414 examples<br />

## Preprocessing:
* Lowercasing, trimming, removing non-alphanumeric characters
* Tokenizing using NLTK

## General Hyperparamters:
* learning_rate: 0.001
* max_gradient_norm: 5.0
* dropout: 0.15
* batch_size: 100
* hidden_size: 200
* context_len: 600 ("The maximum body length of your model")
* question_len: 30 ("The maximum headline length of your model")
* embedding_size: 100 ("Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")

These values are changed on the basis of attention mechanism used and type of reduction:
* attention_type == 'dot_product' and reduction_type == 'max' => max_gradient_norm = 10.0
* attention_type == 'bidaf' => hidden_size = 120
* attention_type == 'self_attention' => self_attn_zsize = 60, hidden_size = 70

## Results
* Dot-Product attention: 71.37%
* Dot-Product attention with lemmatized input: 71.57%
* Self attention: 72.15%
* BiDAF attention: 72.27%
* <b>MultiHeaded Bert atttention: 89.35%</b>
