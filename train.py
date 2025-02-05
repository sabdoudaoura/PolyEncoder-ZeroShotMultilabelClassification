import torch
from torch import nn
import json
import yaml
#from model import BiEncoderModel
from torch.optim import AdamW
from dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score  # Import for computing F1 score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Load the configuration file
with open("/content/drive/MyDrive/Projet Urchadee/Poly encoder : multilabel classification/config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_path = config["data"]["synthetic_data_path"]
model_name = config["model"]["name"]
max_num_labels = config["model"]["max_num_labels"]
learning_rate = float(config["training"]["learning_rate"])
batch_size = config["training"]["batch_size"]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = config["training"]["epochs"]


tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom collate function
def custom_collate_fn(batch):
    # Separate texts and labels from the batch
    # Faire le padding ici
    texts = [item["text"] for item in batch]
    input_ids_text = torch.stack([item['input_ids'] for item in texts]).squeeze(1)
    attention_mask_text = torch.stack([item['attention_mask'] for item in texts]).squeeze(1)


    labels = [item["labels"] for item in batch]
    input_ids_labels = torch.cat([item['input_ids'] for item in labels], dim=0)
    attention_mask_labels = torch.cat([item['attention_mask'] for item in labels], dim=0)

    target_labels = [item["target_labels"] for item in batch]
    target_labels = torch.stack(target_labels, dim=0)


    num_labels = [item["dimension"] for item in batch]
    num_labels = torch.tensor(num_labels).to(device)

    return input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, target_labels, num_labels


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():

        for batch_idx, (input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, target_labels, num_labels) in enumerate(dataloader):
            # Ensure inputs are moved to device (handle dictionaries appropriately)

            input_ids_text = input_ids_text.to(device)
            attention_mask_text = attention_mask_text.to(device)
            input_ids_labels = input_ids_labels.to(device)
            attention_mask_labels = attention_mask_labels.to(device)
            target_labels = target_labels.to(device)
            num_labels = num_labels.to(device)

            scores, mask = model.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, num_labels)
            predictions = (torch.sigmoid(scores) > 0.5).float()

            valid_preds = predictions[mask.bool()]
            valid_targets = target_labels[mask.bool()]

            all_preds.append(valid_preds.cpu())
            all_targets.append(valid_targets.cpu())


    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute F1 score
    f1 = f1_score(all_targets, all_preds, average="micro")
    return f1


if __name__ == "__main__":



    model = PolyEncoderModel(model_name, max_num_labels).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='none') # multiclass classification

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate)

    with open(data_path, 'r', encoding='utf-8') as f:
      train_data = json.load(f)

    train_set, validation_set = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=True)


    # Training loop
    train_dataset = CustomDataset(train_set, model_name, max_num_labels)
    # Training loop
    validation_dataset = CustomDataset(validation_set, model_name, max_num_labels)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    validation_scores = []
    loss_scores = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, target_labels, num_labels) in enumerate(train_loader):


            input_ids_text = input_ids_text.to(device)
            attention_mask_text = attention_mask_text.to(device)
            input_ids_labels = input_ids_labels.to(device)
            attention_mask_labels = attention_mask_labels.to(device)
            target_labels = target_labels.to(device)
            num_labels = num_labels.to(device)
            #Forward pass
            optimizer.zero_grad()
            scores, mask = model.forward(input_ids_text, attention_mask_text, input_ids_labels, attention_mask_labels, num_labels)  # Scores: [B, max_num_labels], mask: [B, max_num_labels]

            # Compute the loss
            loss = criterion(scores, target_labels)
            #multiplier par mask
            loss = (loss * mask).sum()

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Log de la perte
            running_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")

        ## Evaluation
        f1 = evaluate(model, validation_loader, device) #f1_score(all_targets, all_preds, average="micro")
        print(f"Epoch [{epoch + 1}/{epochs}] Evaluation F1 Score: {f1:.4f}")
        loss_scores.append(running_loss / (batch_idx + 1))
        validation_scores.append(f1)