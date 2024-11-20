import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
import os
from copy import deepcopy
import sys
import argparse
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from itertools import product 
import random

# inspirings from https://github.com/balevinstein/Probes

# Set the seed for reproducibility
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ProbeNN(nn.Module):
    """ Neural network for probing the embeddings """
    def __init__(self, input_dim):
        super(ProbeNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x
    

class TrainProbe:
    def __init__(self, dataset_names, layer, probe_method, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df_train = pd.read_pickle(f"{dataset_names[0]}.pkl")
        self.train_embeddings = torch.tensor(self.df_train[f'embeddings{layer}_{probe_method}'].tolist(), dtype=torch.float32).to(self.device)
        self.train_labels = torch.tensor(self.df_train[f'label_{probe_method}'].values, dtype=torch.float32).to(self.device)

        self.df_dev = pd.read_pickle(f"{dataset_names[1]}.pkl")
        self.dev_embeddings = torch.tensor(self.df_dev[f'embeddings{layer}_{probe_method}'].tolist(), dtype=torch.float32).to(self.device)
        self.dev_labels = torch.tensor(self.df_dev[f'label_{probe_method}'].values, dtype=torch.float32).to(self.device)

        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.epochs = hyperparameters["max_epochs"]
        self.patience = hyperparameters["patience"]
        self.accuracy_threshold = hyperparameters["accuracy_threshold"]

        
        self.dataset = TensorDataset(self.train_embeddings, self.train_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        
        
        

    def train_model(self):
        self.criterion = nn.BCELoss()
        self.model = ProbeNN(self.train_embeddings.shape[1]).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        train_losses = []
        dev_losses = []
        early_stopping_triggered = False

    
        best_dev_loss = float('inf') 
        epochs_no_improve = 0
        best_epoch = 0  
        best_model_state = None
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_embeddings, batch_labels in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_embeddings).squeeze()
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(self.dataloader)
            train_losses.append(avg_train_loss)
            self.model.eval()
            with torch.no_grad():
                dev_outputs = self.model(self.dev_embeddings).squeeze()
                dev_loss = self.criterion(dev_outputs, self.dev_labels).item()
                dev_losses.append(dev_loss)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, dev Loss: {dev_loss:.4f}")

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch + 1 
                epochs_no_improve = 0
                best_model_state = deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best dev loss was at epoch: {best_epoch}")
                early_stopping_triggered = True
                break
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self.model, train_losses, dev_losses, best_dev_loss, best_epoch, early_stopping_triggered
    
    def compute_roc_curve(self, dev_labels, dev_pred_prob):
        fpr, tpr, _ = roc_curve(dev_labels, dev_pred_prob)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    
    def evaluate_model(self, best_model):
        best_model.eval()
        with torch.no_grad():
            outputs = best_model(self.dev_embeddings).squeeze()
            predicted = (outputs > self.accuracy_threshold).float()
            accuracy = (predicted == self.dev_labels).float().mean().item()
            loss = nn.BCELoss()(outputs, self.dev_labels).item()
            predicted_probabilities = outputs.cpu().numpy()
            roc_auc = self.compute_roc_curve(self.dev_labels.cpu().numpy(), predicted_probabilities)

        return loss, accuracy, roc_auc



if __name__ == "__main__":
    df_hyperparameters = pd.DataFrame(columns=["model_name", "dataset_name", "layer", "batch_size", 
                                               "learning_rate", "epochs", "accuracy", "auc_roc", 
                                               "best_epoch", "best_dev_loss"])
    datasets = ["fever", "hover"] 
    #layers = [-1, -4, -8, -16, -24]
    layers = [1]
    batch_sizes = [32, 64, 128]
    learning_rates = [0.001, 0.01, 0.05]
    save_probes = True
    train_rounds = 3

    for dataset in datasets:
        for layer in layers:
            print(f"Training probes for dataset: {dataset}, layer: {layer}")
            for probe_method in ["mini_fact", "sentence"]:

                input_path = f"processed_datasets_llama_{dataset}_layer{layer}"
                model_output_path = "probes"  
                probe_dataset_names = [f"{input_path}/{probe_method}_{dataset}_train", f"{input_path}/{probe_method}_{dataset}_dev"]
                
                global_best_dev_loss = float('inf') 
                global_best_model = None
                global_best_hyperparams = None
                global_best_epoch = 0  

                hyperparameter_combinations = list(product(batch_sizes, learning_rates))
                for batch_size, learning_rate in hyperparameter_combinations:
                    print(f"Training with batch_size={batch_size}, learning_rate={learning_rate}")
                    hyperparameters = {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "max_epochs": 100,
                        "accuracy_threshold": 0.5,
                        "patience": 10
                    }
                    probe = TrainProbe(probe_dataset_names, layer, probe_method, hyperparameters)
                    for i in range(train_rounds):
                        set_seed(42 + i)
                        model, train_losses, dev_losses, best_dev_loss, \
                            best_epoch, early_stopping_triggered = probe.train_model()

                        if best_dev_loss < global_best_dev_loss:
                            global_best_dev_loss = best_dev_loss
                            global_best_model = deepcopy(model)
                            global_best_hyperparams = (batch_size, learning_rate)
                            global_best_epoch = best_epoch  
                            print(f"New global best dev loss: {global_best_dev_loss}")
                        
                if global_best_model is not None:
                    loss, accuracy, auc_roc = probe.evaluate_model(global_best_model)
                    print(f"Final Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}")
                    print(f"Best hyperparameters - Batch Size: {global_best_hyperparams[0]}, Learning Rate: {global_best_hyperparams[1]}")
                    print(f"Number of epochs for best model: {global_best_epoch}")

                    df_hyperparameters = pd.concat([
                        df_hyperparameters if not df_hyperparameters.empty else None,
                        pd.DataFrame([{
                            "probe_method": probe_method,
                            "dataset_name": probe_dataset_names[0],
                            "layer": layer,
                            "batch_size": global_best_hyperparams[0],
                            "learning_rate": global_best_hyperparams[1],
                            "accuracy": accuracy,
                            "auc_roc": auc_roc,
                            "best_epoch": global_best_epoch,
                            "best_dev_loss": global_best_dev_loss,
                            "max_epochs": hyperparameters["max_epochs"],
                            "early_stopping": early_stopping_triggered
                        }])
                    ], ignore_index=True)

                    if save_probes:
                        try:
                            if not os.path.exists(model_output_path):
                                os.makedirs(model_output_path)
                            model_save_name = f"{model_output_path}/{probe_method}_embeddings{layer}_{dataset}.pth"
                            torch.save(global_best_model.state_dict(), model_save_name)
                            print(f"Model saved to {model_save_name}")
                        except Exception as e:
                            print(f"Error saving the model: {e}")
    #df_hyperparameters.to_csv(f"{model_output_path}/hyperparameters.csv", index=False)


                       
                


