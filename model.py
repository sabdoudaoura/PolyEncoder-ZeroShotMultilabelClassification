import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolyEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels):
        super(PolyEncoderModel, self).__init__()
        # Shared encoder for both text and labels
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels

    def dot_attention(self, q, k, v, v_mask=None, dropout=None):

        """
        Perform dot-product attention using PyTorch.

        Parameters:
          query (torch.Tensor): shape (batch_size, seq_len_q, d_k)
          key (torch.Tensor): shape (batch_size, seq_len_k, d_k)
          value (torch.Tensor): shape (batch_size, seq_len_v, d_v)
          mask (torch.Tensor, optional): shape (batch_size, seq_len_q, seq_len_k)
          dropout_fn (callable, optional): dropout function to apply on attention weights.

        Returns:
        output (torch.Tensor): shape (batch_size, seq_len_q, d_v)
        attention_weights (torch.Tensor): shape (batch_size, seq_len_q, seq_len_k)
        """

        attention_weights = torch.matmul(q, k.transpose(-1, -2))

        if v_mask is not None:
          extended_v_mask = (1.0 - v_mask.unsqueeze(1)) * -100000.0
          attention_weights += extended_v_mask
        attention_weights = F.softmax(attention_weights, -1)
        if dropout is not None:
          attention_weights = dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output

    def labels_encoder(self, input_ids, attention_mask):
        """
        Encodes a list of texts or labels using the shared encoder.
        """
        outputs = self.shared_encoder(input_ids = input_ids, attention_mask = attention_mask)
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        att_mask = attention_mask.unsqueeze(-1)

        return (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1) ### Global representation for label or sentence

    def text_encoder(self, input_ids, attention_mask, label_embeddings, attention_mask_labels, label_counts):
        """
        Encodes a list of texts or labels using the shared encoder.
        """
        outputs = self.shared_encoder(input_ids = input_ids, attention_mask = attention_mask)
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        att_mask = attention_mask.unsqueeze(-1) ### Create a dimention [B, seq_len, 1] for easier broadcasting
        text_embeddings = outputs.last_hidden_state * att_mask

        augmented_text_embeddings = torch.repeat_interleave(text_embeddings, label_counts, dim=0)

        # No need for mask as all labels and texts in input can be considered
        # result : [B, seq_len , D]
        weighted_text_embeddings = self.dot_attention(label_embeddings.unsqueeze(1), augmented_text_embeddings, augmented_text_embeddings)

        return weighted_text_embeddings

    def forward(self, input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, label_counts):
        """
        texts: List of input texts with batch size B
        batch_labels: List of lists containing labels for each text
        """
        B = input_ids_text.shape[0]

        # Flatten all labels in the batch
        # Shape: [num_of_labels, D]
        label_embeddings = self.labels_encoder(input_ids_labels, attention_mask_labels)

        # Encode texts
        # Shape: [num_of_labels, D]
        text_embeddings = self.text_encoder(input_ids_text, attention_mask_text, label_embeddings, attention_mask_labels, label_counts)

        # Prepare to recover batch structure
        max_num_label = self.max_num_labels
        padded_label_embeddings = torch.zeros(B, max_num_label, label_embeddings.size(-1)).to(device) #Shape [B, max_num_label, D]
        merged_text_embeddings = torch.zeros(B, 1, text_embeddings.size(-1)).to(device) #Shape [B, 1, D]
        mask = torch.zeros(B, max_num_label, dtype=torch.bool).to(device) #Shape [B, max_num_label]

        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                padded_label_embeddings[i, :count, :] = label_embeddings[current:end]
                merged_text_embeddings[i, :, : ] = text_embeddings[current:end, : , :].mean(dim=0)
                mask[i, :count] = 1
                current = end

        # Compute similarity scores between text and each label
        # Each sentence [B, D] -> [B, D, 1] x [B, max_num_label, D]
        # text_embeddings: [B, 1, D]
        # padded_label_embeddings: [B, max_num_label, D]
        # scores: [B, max_num_label]
        scores = torch.bmm(padded_label_embeddings, merged_text_embeddings.transpose(-1, -2)).squeeze(2)

        return scores, mask

    @torch.no_grad()
    def forward_predict(self, texts, batch_labels):
        """
        texts: List of input texts
        labels: List of labels corresponding to the texts
        Returns:
            List of JSON objects with label scores for each text
        """

        texts_tokenized = [self.tokenizer(entry, return_tensors='pt', padding='max_length', truncation=True, max_length=15) for entry in texts]
        batch_labels_tokenized = [self.tokenizer(entry, return_tensors='pt', padding='max_length', truncation=True, max_length=5) for entry in batch_labels]

        input_ids_text = torch.stack([item['input_ids'] for item in texts_tokenized]).squeeze(1).to(device)
        attention_mask_text = torch.stack([item['attention_mask'] for item in texts_tokenized]).squeeze(1).to(device)

        input_ids_labels = torch.cat([item['input_ids'] for item in batch_labels_tokenized], dim=0).to(device)
        attention_mask_labels = torch.cat([item['attention_mask'] for item in batch_labels_tokenized], dim=0).to(device)

        label_counts = torch.tensor([len(labels) for labels in batch_labels])

        scores, mask = self.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, label_counts)
        scores = torch.sigmoid(scores)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(batch_labels[i]):
                if mask[i, j]:
                    text_result[label] = float(f"{scores[i, j].item():.2f}")
            results.append({"text": text, "scores": text_result})
        return results

# # Example Usage
# if __name__ == "__main__":
#     model_name = "bert-base-uncased"
#     max_num_labels = 4
#     model = PolyEncoderModel(model_name, max_num_labels)

#     texts = ["A celebrity chef has opened a new restaurant specializing in vegan cuisine.",
#          "Doctors are warning about the rise in flu cases this season.",
#          "The United States has announced plans to build a wall on its border with Mexico."]
#     batch_labels = [
#         ["Food", "Business", "Politics"],
#         ["Health", "Food", "Public Health"],
#         ["Immigration", "Religion", "National Security"]
#     ]

#     # Prediction with JSON output
#     predictions = model.forward_predict(texts, batch_labels)
#     print("Predictions:", predictions)