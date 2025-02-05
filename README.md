# Poly Encoder - Zero Shot Classification

Our model is a Multilabel Classification Model designed to classify texts into various predefined categories using a bidirectional transformer encoder (BERT-like). By leveraging the Poly encoder architecture, the model effectively considers the interaction between the text and the labels while beein gvery fast at inference. This approach demonstrates superior performance compared to the previous bi-encoder architecture and superior inference speed compared to cross encoder. Consequently, it offers a practical alternative to Large Language Models (LLMs), which, despite their versatility, are often too costly and large for resource-constrained environments. 

<p align="center">
    <a href="https://huggingface.co/sabdou/poly-encoder-model">🤗 Available pretrained model</a>
    <a href="https://arxiv.org/pdf/1905.01969">📜 Original architecture</a>
    </a>
</p>

### Usage
```python

from model import PolyEncoderModel

texts = [
    "The wildlife conservation program is focused on protecting endangered species in Africa.",
    "The government announced a new initiative to combat poverty in rural areas.",
    "A celebrity chef has opened a new restaurant specializing in vegan cuisine."
    ]
batch_labels = [

        ["Conservation", "Business", "Animals", "Africa"],
        ["Politics", "Social Issues", "Economy", "Technolgy"],
        ["Food", "Business","Politics", "Vegan"]

    ]
# Load the model
model = CrossEncoderModel("sabdou/poly-encoder-model", max_num_labels=6)
# Prediction with JSON output
predictions = model.forward_predict(texts, batch_labels)
print("Predictions:", predictions)

```


#### Expected Output

```
Predictions: [
    {'text': 'The wildlife conservation program is focused on protecting endangered species in Africa.',
  'scores': {'Conservation': 1.0,
             'Business': 0.0,
             'Animals': 1.0,
             'Africa': 0.99}},

 {'text': 'The government announced a new initiative to combat poverty in rural areas.',
  'scores': {'Politics': 1.0,
             'Social Issues': 0.99,
             'Economy': 1.0,
             'Technolgy': 0.0}},

 {'text': 'A celebrity chef has opened a new restaurant specializing in vegan cuisine.', 
 'scores': {'Food': 1.0,
            'Business': 1.0, 
            'Politics': 0.0, 
            'Vegan': 1.0}}]


```


#### Usecase

This model can be used in

- Augmented user experience : To detect important informations in customer reviews.
- Content moderation: Automatically flagging inappropriate or harmful content in social media posts.
- News categorization: Classifying news articles into predefined categories for better organization and retrieval.
- Sentiment analysis: Identifying the sentiment of customer feedback to improve products and services.
- Document tagging: Automatically tagging documents with relevant keywords for easier search and retrieval.
- Chatbot enhancement: Improving chatbot responses by understanding the context and intent of user queries.

#### Data

Synthetic data generated with gpt4-mini and gemini 

## Author 🧑‍💻
- [Salim ABDOU DAOURA](https://github.com/sabdoudaoura)
