# LLM-Calculator

![https://wandb.ai/kwargs/llmcalc?workspace=user-kwargs](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

🤗 HuggingFace Transformers

# Model choice
So here we have two models - LSTM and BERT
## LSTM
  [Check out code at Colab](https://colab.research.google.com/drive/1Talo7MmE689e-KkSXHqVBSiipnXZEWRJ?pli=1#scrollTo=nAMDbVkOVTjA&uniqifier=1)
  
  First model that came at my mind is LSTM, because it's fairly simple model that performs well at this type of tasks
  
  Architecture is following:
  - Input is tokenized/"vectorized" as e.g "2+2" -> [2, 10, 2]
  - Embedding layer
  - LSTM
  - LayerNorm
  - Linear
  - ReLU
  - Linear
  
  > Note on LayerNorm - it was better to use BatchNorm to make LSTM more stable, but LayerNorm was easier to implement and worked just fine
  
  The only downside of this solution is that it solves regression task, instead of honest text2text

  Some attempts were made to solve it as text2text, but I've bailed on it

  Nevertheless, I can say a few words about possible solution:
  - CosineEmbeddingLoss works very well in this task
      > Because we use *vectorized* representations of numbers (e.g. 42 -> [4, 2]), we can't really use L1Loss or MSELoss, but we can use vector-specific losses such as cosine one
  - Reversing input gives some boost to score

## BERT
[Check out code at Colab](https://colab.research.google.com/drive/1TQ8qUq3Or4U-hAdy8iOpkxRSdnWLKIia?usp=sharing)

Very simple yet powerful, just tokenize and pass to Trainer to finetune. Actually, I should did it before, because gives incredible resuls with minimal work -  near zero test loss for only couple of epochs!

Here I solve this task as Masked Language Modeling task so my input looks like `2+2=[MASK]`


# Evaluation
Input (LSTM) `2+2`

Input (BERT) `2+2=[MASK]`

Output `4`

## Experiments
- [x] BERT is better than LSTM (as expected)
- [x] I've evaluated model on different lenghts of inputs - left and right
  - Longer input - worse score (possible underfit due to increasing data size as model remains the same)
  - Model performs better when left input is longer than right
- [x] Out of domain testing (e.g. train on 2 digit numbers, evaluate on 3 digit numbers)

Full results are on [Weights&Biases](https://wandb.ai/kwargs/llmcalc?workspace=user-kwargs)

## Metrics
* MAE/MSE - obvious regression metrics
  * ~0.02 for BERT
  * ~0.8 for best performing LSTM
  * Full results on LSTM on different input lengths
  * | left\right | 1     | 2      | 3      |
    |------------|-------|--------|--------|
    | 1          | 1.451 | 24.946 | 20.295 |
    | 2          | 0.849 | 1.11   | 36.795 |
    | 3          | 1.926 | 1.982  | x      |
* Cosine loss - if you don't care about whole number rather then digits separately
* Accuracy/Precision/Recall/F1 - if we only care, guessed we number correctly or not
  * Pretty low even for BERT, because it's only off by one or so, but it's still considered as wrong guess

## Conclusions
Let's take BERT as our final model because it performs the best compared to LSTM

\+ It's small - only 110M params!
  * It was allowed to use <4B models, so theoretically BERT can perform even better

\+ Can take relatively big input - I've tried 512 max (two ~200 digits numbers) 

\- It can accept long inputs, but it performs worse on it

# Relevant papers/materials
* http://arxiv.org/abs/1410.4615
  * Paper about my own literal problem. Authors show that it is possible for LSTM to add two 9-digit numbers with 99%-accuracy
  * Main topic of paper is empirically evaluating the expressiveness and the learnability of LSTMs in the sequence-to-sequence regime by training them to evaluate short computer programs, a domain that has traditionally been seen as too complex for neural networks.
  * Also this paper and one below shows that input of LSTM may be optionally be reversed, which was shown to increase performance in many tasks.
* http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
  * The paper presents a general end-to-end approach to sequence learning using multilayered LSTM to map the input sequence to a vector of fixed dimensionality, and another deep LSTM to decode the target sequence from the vector.
  * On an English to French translation task from the WMT-14 dataset, the LSTM achieved a BLEU score of 34.8 on the entire test set, compared to a phrase-based SMT system's score of 33.3 on the same dataset.
  * The LSTM learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice.
  * Reversing the order of the words in all source sentences (but not target sentences) improved the LSTM's performance markedly.
* https://arxiv.org/abs/1909.07940
  8 Most NLP models treat numbers in text as other tokens, embedding them as distributed vectors.
  * Investigating the numerical reasoning capabilities of a state-of-the-art question answering model on the DROP dataset, the model excels on questions that require numerical reasoning, indicating it already captures numeracy.
  * Token embedding methods (e.g., BERT, GloVe) were probed on synthetic list maximum, number decoding, and addition tasks to understand how this capability emerges.
  * Standard embeddings such as GloVe and word2vec accurately encode magnitude for numbers up to 1,000.
  * Character-level embeddings are even more precise, with ELMo capturing numeracy the best among pre-trained methods.
  * BERT, which uses sub-word units, is less exact in capturing numeracy.
* https://towardsdatascience.com/nlp-numeracy-pt-2-embeddings-language-models-and-calculators-615a346737c2
  * Intro article on numeracy in language models
* https://keras.io/examples/nlp/addition_rnn/
  * Literal solution of my assigment on Keras using LSTM 
