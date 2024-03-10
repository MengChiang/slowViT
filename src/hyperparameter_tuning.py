import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from models import SlowViT
from dataset import ActionRecognitionDataset
from load_models import load_models
import configparser
import wandb

def get_api_key_from_config(config_path):
    # Create a config parser
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(config_path)

    # Get the API key
    return config.get('WANDB', 'api_key')

def train_model_for_one_epoch(model, data_loader, optimizer, loss_function):
    for batch in data_loader:
        video, text, labels = batch
        optimizer.zero_grad()
        outputs = model(video, text)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, data_loader):
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            video, text, labels = batch
            outputs = model(video, text)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            actual_labels.extend(labels.tolist())
    return actual_labels, predictions

def main():
    # Log in to W&B
    api_key = get_api_key_from_config('config.ini')
    wandb.login(key=api_key)

    # Initialize W&B project
    wandb.init(project="my_project")

    # Define your tokenizer and num_epochs
    tokenizer = None  # TODO: Initialize your tokenizer
    num_epochs = 10  # TODO: Set the number of epochs you want to train for

    # Load your dataset
    your_dataset = ActionRecognitionDataset('annotations.csv', 'videos', tokenizer)

    # Initialize the model
    slowfast_model, bert_model = load_models()
    combined_model = SlowViT(slowfast_model, bert_model)

    # Set up the loss function
    loss_function = nn.CrossEntropyLoss()

    # Define the parameters to tune
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
    }

    # Create a grid of parameters
    grid = ParameterGrid(param_grid)

    # For each combination of parameters
    for params in grid:
        # Start a new run
        run = wandb.init(project="slow_ViT", config=params)

        # Update the model parameters
        optimizer = Adam(combined_model.parameters(), lr=params['learning_rate'])
        data_loader = DataLoader(your_dataset, batch_size=params['batch_size'], shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            train_model_for_one_epoch(combined_model, data_loader, optimizer, loss_function)

        # Set up DataLoader for test data
        test_dataset = None  # TODO: Set up your test dataset
        test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

        # Run the model on the test data and collect the predictions and actual labels
        actual_labels, predictions = evaluate_model(combined_model, test_data_loader)

        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, average='weighted')
        recall = recall_score(actual_labels, predictions, average='weighted')
        f1 = f1_score(actual_labels, predictions, average='weighted')

        # Log metrics to W&B
        wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

        # Finish the run
        run.finish()

if __name__ == "__main__":
    main()