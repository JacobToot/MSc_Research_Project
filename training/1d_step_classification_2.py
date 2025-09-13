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

from utils.dataset_1d_ext import DataSet1D
from models.classifier_1d import Conv1DClassifier
from simulations.quasi_crystal_1d import quasi_crystal_1d
from simulations.random_step_1d import random_step_1d
from simulations.gaussian_step_1d import gaussian_step_1d
from utils.permutation import permutation_1d
from utils.generate_irrational import generate_irrational

dataset = DataSet1D()
seq_len = 2500
dataset = dataset.quasi_crystal(number_of_sequences=3000, lattice_spacing=1, slope=lambda: generate_irrational(upper_limit=10), acceptance_window=lambda: np.random.randint(2,20), number_of_points=seq_len)
print(1)
dataset = dataset.permutation(number_of_pairs=int(0.4 * seq_len), permute_label=1)
print(2)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

num_classes = len(set(dataset.labels))

model = Conv1DClassifier(seq_len=seq_len, classes = num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

# model = Conv1DClassifier(seq_len=seq_len, classes = num_classes+1)
# weights = torch.ones(num_classes + 1) 
# weights[num_classes] = 0.5
# criterion = nn.CrossEntropyLoss(weight=weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 100

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
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_outputs = model(x_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += y_val.size(0)
            val_correct += (val_predicted == y_val).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

torch.save(model.state_dict(), os.path.join(weights_path, '1d_step_classifier.pth'))

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for x_val, y_val in val_loader:
        val_outputs = model(x_val)
        _, val_predicted = torch.max(val_outputs.data, 1)
        all_preds.extend(val_predicted.cpu().numpy())
        all_labels.extend(y_val.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Quasi-Crystal', 'Random Step', 'Gaussian Step'], yticklabels=['Quasi-Crystal', 'Random Step', 'Gaussian Step'])
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_path, 'confusion_matrix_2.png'))