import random
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class CustomDataset(Dataset):
    def __init__(self, data, model_name, max_num_labels, transform=None, max_num_negatives=2):

        # Open and read json
        self.data = data
        self.max_num_negatives = max_num_negatives
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transform = transform
        self.all_labels = list(set([label for labels in [element["labels"] for element in data] for label in labels]))
        self.max_num_labels = max_num_labels

    def __getitem__(self, index):

      #meilleure pratique, ne pas faire de padding ici, torch.nn.functional pour le padding
      negative_labels = self.negative_sampling([self.data[index]['labels']], self.all_labels)
      self.texts = self.tokenizer(self.data[index]['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=15)
      self.labels = self.tokenizer(self.data[index]['labels'] + negative_labels[0], return_tensors='pt', padding='max_length', truncation=True, max_length=5)
 
      p = len(self.data[index]['labels'])
      q = len(negative_labels[0])
      
      assert self.max_num_labels >= (p+q), "maximum number of labels should be >= to p+q"
      target_labels = torch.cat([torch.ones(p), torch.zeros(q), torch.zeros(self.max_num_labels -(p+q))]) #Ensures that scores and target are the same length

      return {"text" : self.texts,
              "labels" :  self.labels,
              "target_labels" : target_labels,
              "dimension" : p+q}

    def __len__(self):
        return len(self.data)

    def negative_sampling(self, batch_labels, all_labels):
        num_negatives = random.randint(1, self.max_num_negatives)
        negative_samples = []
        for labels in batch_labels:
            neg = random.sample([l for l in all_labels if l not in labels], num_negatives)
            negative_samples.append(neg)
        return negative_samples

