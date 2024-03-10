import nni
from models import SlowViT
from load_models import load_models
import torch
import torch.nn as nn
import torchvision.models as models
from nni.nas.pytorch.mutables import LayerChoice
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import ActionRecognitionDataset

class SearchableSlowFast(nn.Module):
    def __init__(self):
        super(SearchableSlowFast, self).__init__()
        self.layer_choice = LayerChoice([
            models.resnet18(pretrained=True),
            models.resnet34(pretrained=True),
            models.resnet50(pretrained=True)
        ])

    def forward(self, x):
        return self.layer_choice(x)

def main(params):
    # Define your tokenizer and num_epochs
    tokenizer = None  # TODO: Initialize your tokenizer
    num_epochs = params['num_epochs']

    # Load your dataset
    your_dataset = ActionRecognitionDataset('annotations.csv', 'videos', tokenizer)

    # Initialize the model
    slowfast_model = SearchableSlowFast()
    bert_model = load_models()
    combined_model = SlowViT(slowfast_model, bert_model)

    # Set up the loss function
    loss_function = nn.CrossEntropyLoss()

    # Define the parameters
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    # Update the model parameters
    optimizer = Adam(combined_model.parameters(), lr=learning_rate)
    data_loader = DataLoader(your_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            video, text, labels = batch
            optimizer.zero_grad()
            outputs = combined_model(video, text)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    # Set up DataLoader for test data
    test_dataset = None  # TODO: Set up your test dataset
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Run the model on the test data and collect the predictions and actual labels
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in test_data_loader:
            video, text, labels = batch
            outputs = combined_model(video, text)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            actual_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, average='weighted')
    recall = recall_score(actual_labels, predictions, average='weighted')
    f1 = f1_score(actual_labels, predictions, average='weighted')

    # Return the performance
    return f1

if __name__ == '__main__':
    params = nni.get_next_parameter()
    f1 = main(params)
    nni.report_final_result(f1)