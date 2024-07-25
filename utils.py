import os
import requests
from zipfile import ZipFile
import shutil
import torchaudio
from pathlib import Path
import glob
from torchaudio import transforms
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from config import MODELS_PATH,FIG_PATH


class AudioDataProcessor:
    def __init__(self, datapath):
        self.datapath = datapath
        self.file_list = self._get_file_list()
        
    def _get_file_list(self):
        # Use glob to fetch list of all .wav file paths recursively
        file_list = glob.glob(os.path.join(self.datapath, '**', '*.wav'), recursive=True)
        return file_list
    
    def _load_audio(self, filepath):
        # Load audio file
        waveform, sample_rate = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sample_rate

    def _generate_log_spectrogram(self, waveform, sample_rate):
        # Create a spectrogram
        spectrogram = transforms.Spectrogram(n_fft=512, hop_length=256, power=2.)(waveform)
        # Convert to log scale
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        # log_spectrogram = torch.log1p(spectrogram)  # torch.log1p is log(1 + x) to avoid log(0)
        return log_spectrogram

    def __getitem__(self, index):
        if index >= len(self.file_list):
            raise IndexError("Index out of range")
        
        filepath = self.file_list[index]
        waveform, sample_rate = self._load_audio(filepath)
        log_spectrogram = self._generate_log_spectrogram(waveform, sample_rate)
        
        return {
            "filepath": filepath,
            "waveform": (waveform, sample_rate),
            "log_spectrogram": log_spectrogram
        }

    @staticmethod
    def get_samples(processor, num_samples = 6):
        samples = []
        classes = []
        for i in range(len(processor.file_list)):
            item = processor[i]
            if Path(item['filepath']).parent.name in classes:
                continue
            else:
                item['class'] = Path(item['filepath']).parent.name
                samples.append(item)
                classes.append(Path( item['filepath']).parent.name)
            if len(samples) == num_samples:
                break
        return samples

def train_model(model, train_loader, val_loader, criterion, optimizer,scheduler,early_stopping, num_epochs=25, device="cpu", save_path='saved_model.pth'):
    """
    Train the model with the given data loaders, loss function, and optimizer.

    Parameters:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        optimizer (torch.optim.Optimizer): Optimizer.
        early_stopping (EarlyStopping): Early stopping object.
        num_epochs (int): Number of epochs to train the model.
        device (str): Device to use for training ('cpu' or 'cuda').
        save_path (str): Path to save the best model.

    Returns:
        model (nn.Module): The trained model.
        dict: Dictionary containing training and validation loss and accuracy history.
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    min_valid_loss = np.inf

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = 0.0
        train_corrects = 0
        model.train()  # Set model to training mode
        
        for samples, labels in train_loader:
            data, labels = samples["data"].to(device), labels.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate training loss and accuracy
            train_loss += loss.item() * data.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects / len(train_loader.dataset)
        
        valid_loss = 0.0
        valid_corrects = 0
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            for samples, labels in val_loader:
                data, labels = samples['data'].to(device), labels.to(device)
                
                # Forward pass
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Calculate validation loss and accuracy
                valid_loss += loss.item() * data.size(0)
                valid_corrects += torch.sum(preds == labels.data).item()
        
        valid_loss /= len(val_loader.dataset)
        valid_acc = valid_corrects / len(val_loader.dataset)
        
        print(f'Training Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Validation Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')
        
        if valid_loss < min_valid_loss:
            print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{MODELS_PATH}/{save_path}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)
        
        if early_stopping(valid_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        # Step the scheduler
        scheduler.step()

    return model, history


def test_model(model, test_loader, criterion, device="cpu"):
    test_loss = 0.0
    test_corrects = 0
    all_labels = []
    all_preds = []
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data['data'].to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Calculate test loss and accuracy
            test_loss += loss.item() * data.size(0)
            test_corrects += torch.sum(preds == labels.data).item()
            
            # Collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, class_names, modelstr="resnet18"):
    cm = confusion_matrix(true_labels, pred_labels)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{FIG_PATH}/{modelstr}_confusion_matrix.png')
    plt.close()
    # plt.show()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False



def sample_balanced_representations(features, labels, classes, samples_per_class=10):
    """
    Sample a balanced set of representations from each class.

    Args:
    features (np.array): 2D array of feature vectors
    labels (np.array): 1D array of labels corresponding to each feature vector
    classes (list): List of all class names
    samples_per_class (int): Number of samples to select from each class

    Returns:
    tuple: (selected_representations, selected_labels)
        selected_representations (np.array): 2D array of selected feature vectors
        selected_labels (np.array): 1D array of labels for selected features
    """
    selected_representations = []
    selected_labels = []

    for class_name in classes:
        # Find indices of samples belonging to the current class
        class_indices = np.where(np.array(labels) == class_name)[0]
        
        # Determine how many samples to select (in case there are fewer than requested)
        n_samples = min(samples_per_class, len(class_indices))
        
        # Randomly select indices if there are more samples than needed
        if len(class_indices) > n_samples:
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        else:
            selected_indices = class_indices

        # Add selected samples and labels to the lists
        selected_representations.extend(features[selected_indices])
        selected_labels.extend([class_name] * n_samples)

    return np.array(selected_representations), np.array(selected_labels)

def compute_cosine_similarity(features):
    """
    Compute the cosine similarity matrix for the given features.
    
    Args:
    features (np.array): 2D array of feature vectors

    Returns:
    np.array: 2D array of cosine similarities
    """
    return cosine_similarity(features)

def plot_similarity_heatmap(similarity_matrix, labels, output_path, use_mask=False):
    """
    Plot a heatmap of the similarity matrix with improved tick labels.
    
    Args:
    similarity_matrix (np.array): 2D array of similarity scores
    labels (np.array): 1D array of labels corresponding to each feature vector
    output_path (str): Path to save the output image
    use_mask (bool): Whether to use a mask to show only the lower triangle
    """
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.0)
    
    # Create a mask to show only the lower triangle of the heatmap
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    
    # Get unique labels and their first occurrences
    unique_labels, unique_indices = np.unique(labels, return_index=True)
    if use_mask:
        ax = sns.heatmap(similarity_matrix, cmap='viridis', mask=mask,
                         xticklabels=False, yticklabels=False,
                         cbar_kws={"shrink": .8})
    else:
        # Plot the heatmap
        ax = sns.heatmap(similarity_matrix, cmap='viridis', 
                     xticklabels=False, yticklabels=False,
                     cbar_kws={"shrink": .8})
    
    # Set tick locations and labels
    tick_locations = unique_indices + 0.5  # Center ticks in the middle of each class
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    ax.set_xticklabels(unique_labels, rotation=90, ha='center')
    ax.set_yticklabels(unique_labels, rotation=0, va='center')
    
    # Improve tick label visibility
    ax.tick_params(axis='both', which='major', labelsize=8, length=0)
    
    # plt.title("Cosine Similarity Heatmap of Feature Representations", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    pass
