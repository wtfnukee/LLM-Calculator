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
  - Input is tokenized as e.g "2+2" -> [2, 10, 2]
  - Embedding layer
  - LSTM
  - LayerNorm
  - Linear
  - ReLU
  - Linear
  
  The only downside of this solution is that it solves regression task, instead of honest text2text

  Some attempts were made to solve it as text2text, but I've bailed on it

  Nevertheless, I can say a few words about possible solution:
  - CosineEmbeddingLoss works very well in this task
      > Because we use *vectorized* representations of numbers (e.g. 42 -> [4, 2]), we can't really use L1Loss or MSELoss, but we can use vector-specific losses such as cosine one
  - Reversing input gives some boost to score

## BERT
[Check out code at Colab](https://colab.research.google.com/drive/1TQ8qUq3Or4U-hAdy8iOpkxRSdnWLKIia?usp=sharing)
Very simple yet powerful, just tokenize and pass to Trainer to finetune. Actually, I should did it before, because gives incredible resuls with minimal work -  near zero test loss for only couple of epochs!

# Train
## LSTM
20 epochs for 6 different dataset sizes

## BERT
10 epochs for one dataset

# Evaluation
Input (LSTM) `2+2`

Input (BERT) `2+2=[MASK]`

Output `4`

## Experiments
- [x] BERT is better than LSTM (as expected)
- [x] I've evaluated model on different lenghts of inputs - left and right
  - Longer input - worse score (possible underfit due to increasing data size as model remains the same)
  - Model performs better when left input is longer than right
- [x] Out of domain testing 

Full results are on [Weights&Biases](https://wandb.ai/kwargs/llmcalc?workspace=user-kwargs)

## Metrics
* MAE/MSE - obvious regression metrics
* Cosine loss - if you don't care about whole number rather then digits separately
* Accuracy/Precision/Recall/F1 - if we only care, guessed we number correctly or not

## Conclusions
Let's take BERT as our final model because it performs the best
+ It's small - only 110M params!
+ Relatively endless input - I've tried 512 max (two ~200 digits numbers) 
+ 

# Relevant papers/materials
* http://arxiv.org/abs/1410.4615
* http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
* https://arxiv.org/abs/1909.07940
* https://towardsdatascience.com/nlp-numeracy-pt-2-embeddings-language-models-and-calculators-615a346737c2
* https://keras.io/examples/nlp/addition_rnn/
* https://habr.com/ru/companies/yandex/articles/493950/ 🥴🥴🥴
