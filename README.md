# LLM-Calculator

![https://wandb.ai/kwargs/llmcalc?workspace=user-kwargs](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)

# Model choice
So here we have two models - LSTM and BERT
## LSTM
  [Check out code at Colab](https://colab.research.google.com/drive/1Talo7MmE689e-KkSXHqVBSiipnXZEWRJ?pli=1#scrollTo=nAMDbVkOVTjA&uniqifier=1)
  
  First model that came at my mind is LSTM, because it's fairly simple model that performs well at this type of tasks
  
  Architecture is following:
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
[Weights&Biases](https://wandb.ai/kwargs/llmcalc?workspace=user-kwargs)

# Evaluation
Input (LSTM) `2+2`

Input (BERT) `2+2=[MASK]`

Output `4`

I've evaluated model on different leng

## Metrics
* MAE/MSE - obvious regression metrics
* Cosine loss - if you don't care about whole number rather then digits separately


# Relevant papers/materials
* http://arxiv.org/abs/1410.4615
* http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
* https://arxiv.org/abs/1909.07940
* https://towardsdatascience.com/nlp-numeracy-pt-2-embeddings-language-models-and-calculators-615a346737c2
* https://keras.io/examples/nlp/addition_rnn/
* https://habr.com/ru/companies/yandex/articles/493950/ 🥴🥴🥴
