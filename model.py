import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional, Callable

class PolyEncoderModel(nn.Module):
    def __init__(self, model_name: str, max_num_labels: int, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shared_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels

    def dot_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_mask: Optional[torch.Tensor] = None,
        dropout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
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
        attention_weights = F.softmax(attention_weights, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output

    def labels_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes a list of labels using the shared encoder.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs with shape [B, seq_len].
            attention_mask (torch.Tensor): Tensor of attention masks with shape [B, seq_len].
        Returns:
            torch.Tensor: Global representation for label or sentence with shape [B, D].
        """

        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        att_mask = attention_mask.unsqueeze(-1).float()
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        pooled = (outputs.last_hidden_state * att_mask).sum(dim=1) / (att_mask.sum(dim=1) + 1e-8)
        return pooled

    def text_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_embeddings: torch.Tensor,
        label_counts: torch.Tensor
    ) -> torch.Tensor:
        """
            Encodes a list of texts or labels using the shared encoder and performs mask-aware pooling.
        Args:
            input_ids (torch.Tensor): Tensor of input token IDs with shape [B, seq_len].
            attention_mask (torch.Tensor): Tensor of attention masks with shape [B, seq_len].
            label_embeddings (torch.Tensor): Tensor of label embeddings with shape [num_labels, D].
            label_counts (torch.Tensor): Tensor containing the count of labels for each input with shape [B,1].
        Returns:
            torch.Tensor: Weighted text embeddings with shape [B, seq_len, D].
        """
        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mask aware pooling
        # last_hidden_state: [B, seq_len, D]
        att_mask = attention_mask.unsqueeze(-1).float()
        text_embeddings = outputs.last_hidden_state * att_mask
        augmented_text_embeddings = torch.repeat_interleave(text_embeddings, label_counts, dim=0)
        # No need for mask as all labels and texts in input can be considered
        # result : [B, seq_len , D]
        weighted_text_embeddings = self.dot_attention(
            label_embeddings.unsqueeze(1), augmented_text_embeddings, augmented_text_embeddings
        )
        return weighted_text_embeddings

    def forward(
        self,
        input_ids_text: torch.Tensor,
        attention_mask_text: torch.Tensor,
        input_ids_labels: torch.Tensor,
        attention_mask_labels: torch.Tensor,
        label_counts: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the PolyEncoderModel.

        Args:
            input_ids_text (torch.Tensor): Tensor of input text token IDs with shape [batch_size, seq_len].
            attention_mask_text (torch.Tensor): Tensor of attention masks for input texts with shape [batch_size, seq_len].
            input_ids_labels (torch.Tensor): Tensor of input label token IDs with shape [batch_size, max_num_label].
            attention_mask_labels (torch.Tensor): Tensor of attention masks for input labels with shape [batch_size, max_num_label].
            label_counts (List[int]): List containing the number of labels for each text in the batch.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - scores (torch.Tensor): Similarity scores between text and each label with shape [batch_size, max_num_label].
                - mask (torch.Tensor): Mask tensor indicating valid labels with shape [batch_size, max_num_label].
        """
        B = input_ids_text.size(0)
        # Flatten all labels in the batch
        # Shape: [num_of_labels, D]
        label_embeddings = self.labels_encoder(input_ids_labels, attention_mask_labels)
        # Encode texts
        # Shape: [num_of_labels, D]
        text_embeddings = self.text_encoder(input_ids_text, attention_mask_text, label_embeddings, label_counts)
        
        # Reconstruction de la structure batch (à optimiser éventuellement pour éviter la boucle)
        padded_label_embeddings = torch.zeros(B, self.max_num_labels, label_embeddings.size(-1), device=self.device)
        merged_text_embeddings = torch.zeros(B, 1, text_embeddings.size(-1), device=self.device)
        mask = torch.zeros(B, self.max_num_labels, dtype=torch.bool, device=self.device)
        
        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                padded_label_embeddings[i, :count, :] = label_embeddings[current:end]
                merged_text_embeddings[i] = text_embeddings[current:end].mean(dim=0, keepdim=True)
                mask[i, :count] = True
                current = end

        # Compute similarity scores between text and each label
        # Each sentence [B, D] -> [B, D, 1] x [B, max_num_label, D]
        # text_embeddings: [B, 1, D]
        # padded_label_embeddings: [B, max_num_label, D]
        # scores: [B, max_num_label]
        scores = torch.bmm(padded_label_embeddings, merged_text_embeddings.transpose(-1, -2)).squeeze(-1)
        return scores, mask

    @torch.no_grad()
    def forward_predict(self, texts: List[str], batch_labels: List[List[str]]) -> List[dict]:
        """
        Perform forward prediction on a batch of texts and their corresponding labels.
        Args:
            texts (List[str]): List of input texts to be classified.
            batch_labels (List[List[str]]): List of lists, where each sublist contains labels corresponding to each text.
            List[Dict[str, Any]]: List of dictionaries, where each dictionary contains the input text and a dictionary of label scores.
        """
        texts_tokenized = [self.tokenizer(text, return_tensors='pt', padding='max_length',
                                            truncation=True, max_length=15) for text in texts]
        batch_labels_tokenized = [self.tokenizer(label, return_tensors='pt', padding='max_length',
                                                   truncation=True, max_length=5) for labels in batch_labels for label in labels]

        input_ids_text = torch.stack([item['input_ids'] for item in texts_tokenized]).squeeze(1).to(self.device)
        attention_mask_text = torch.stack([item['attention_mask'] for item in texts_tokenized]).squeeze(1).to(self.device)
        input_ids_labels = torch.cat([item['input_ids'] for item in batch_labels_tokenized], dim=0).to(self.device)
        attention_mask_labels = torch.cat([item['attention_mask'] for item in batch_labels_tokenized], dim=0).to(self.device)
        label_counts = torch.tensor([len(labels) for labels in batch_labels])

        scores, mask = self.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, label_counts)
        scores = torch.sigmoid(scores)

        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(batch_labels[i]):
                if mask[i, j]:
                    # Utilisation d'une f-string pour formater le score
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