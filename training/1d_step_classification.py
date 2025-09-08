import numpy as np
import random
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(repo_root, "src"))

results_path = os.path.join(repo_root, "results", "output", "1D_Step_Classification")
os.makedirs(results_path, exist_ok=True)

weights_path = os.path.join(repo_root, "results", "weights", "1D_Step_Classification")
os.makedirs(weights_path, exist_ok=True)

from utils.dataset_1d import DistanceDataset
from models.classifier_1d import Conv1DClassifier
from simulations.quasi_crystal_1d import quasi_crystal_1d
from simulations.random_step_1d import random_step_1d
from simulations.gaussian_step_1d import gaussian_step_1d

dataset = DistanceDataset(n_samples=300, seq_len=100)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Conv1DClassifier(seq_len=100)

criterion = nn.CrossEntropyLoss()  # expects logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)            
        loss = criterion(outputs, y_batch)  
        loss.backward()                     
        optimizer.step()                    
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}%")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            outputs = model(x_val)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_val).sum().item()
            val_total += y_val.size(0)
    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for x_val, y_val in val_loader:
        outputs = model(x_val)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_val.tolist())

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
plt.show()
