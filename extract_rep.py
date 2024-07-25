import click
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from dataset import MAudioDataset, SpectrogramDataset
from models import CustomCNNModel, ContrastiveCNN
from config import DATA_PATH, BATCH_SIZE, SEED, MODELS_PATH, RESULTS_PATH, SAMPLING_RATE, LATENT_DIM,FEMALE_DATA_PATH,EXPORT_DATA_PATH,HUMAN_DATA_PATH,CHIMPANZEE_DATA_PATH
from utils import train_model, test_model, EarlyStopping, plot_confusion_matrix

# Set random seed for reproducibility
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('--modelstr', default='resnet18', help='Model architecture to use')
@click.option('--experiment', default=1, type=int, help='Experiment number')
@click.option('--target_class', default='', help='Target class for classification')
@click.option('--model_type', default='classifier', type=click.Choice(['classifier', 'contrastive']), help='Type of model to use')
@click.option('--method', default='triplet', type=click.Choice(['triplet', 'supcon']), help='Type of method to use')
def main(modelstr, experiment, target_class, model_type, method):
    """
    Main function to extract features from audio samples using a pre-trained model.
    """
    print(f"Starting feature extraction with the following parameters:")
    print(f"Model: {modelstr}")
    print(f"Experiment: {experiment}")
    print(f"Target class: {target_class}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Method: {method}")


    if target_class == 'chimpanzee_ir':
        dataset = SpectrogramDataset(DATA_PATH, duration=2, target_sample_rate=SAMPLING_RATE)
    else:
        dataset = SpectrogramDataset(FEMALE_DATA_PATH, duration=2, target_sample_rate=SAMPLING_RATE)
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # Load the appropriate model
    if model_type == "classifier":
        print("Loading classifier model...")
        model = CustomCNNModel(num_classes=11, weights=None, modelstr=modelstr)
        if target_class == 'chimpanzee_ir' or target_class == 'human_ir':
            model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_{modelstr}_{target_class}_experiment_{experiment}.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_{modelstr}_chimpanzee_ir_experiment_{experiment}.pth', map_location=device))
        features_model = torch.nn.Sequential(*list(model.base_model.children())[:-1], torch.nn.Flatten())
    elif model_type == "contrastive":
        print("Loading contrastive model...")
        features_model = ContrastiveCNN(latent_dim=LATENT_DIM, weights=None, modelstr=modelstr)
        print("We are here...")
        # features_model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_type}_trained_model_{modelstr}_experiment_{experiment}.pt', map_location=device))
        if target_class == 'chimpanzee_ir' or target_class == 'human_ir':
            features_model.load_state_dict(torch.load(f'{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{method}_experiment_{experiment}.pt', map_location=device))
        else:
            features_model.load_state_dict(torch.load(f'{MODELS_PATH}/contrastive_trained_model_{modelstr}_chimpanzee_ir_{method}_experiment_{experiment}.pt', map_location=device))

    features_model.to(device)
    features_model.eval()

    features = []
    labels = []

    # Iterate over the dataloader and get the embeddings
    print("Extracting features...")
    for sample, label in tqdm(dataloader, desc="Processing batches"):
        with torch.no_grad():
            sample_data = sample['data'].to(device)
            batch_features = features_model(sample_data).detach().cpu().numpy()
            features.append(batch_features)
            labels.append(label.numpy())

    # Convert to numpy array
    features = np.vstack(features)
    labels = np.hstack(labels)
    label_classes = [dataset.classes[i] for i in labels]

    # Save as compressed numpy array
    if model_type == "classifier":
        if target_class == 'chimpanzee_ir' or target_class == 'human_ir':
            output_file = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_experiment_{experiment}.npz'
        else:
            output_file = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_experiment_{experiment}_females.npz'
    else:
        if target_class == 'chimpanzee_ir' or target_class == 'human_ir':
            output_file = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_{method}_experiment_{experiment}.npz'
        else:
            output_file = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_{method}_experiment_{experiment}_females.npz'
    
    np.savez_compressed(output_file, features=features, labels=labels, label_classes=label_classes)
    print(f'Features saved to {output_file}')

    print(f"Feature extraction completed. Total samples processed: {len(labels)}")

if __name__ == "__main__":
    main()