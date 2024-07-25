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
from torchcam.methods import ScoreCAM,SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import MAudioDataset, SpectrogramDataset
from models import CustomCNNModel, ContrastiveCNN
from config import DATA_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED, MODELS_PATH, RESULTS_PATH, NUM_EXPERIMENTS, MODELSTRS, SAMPLING_RATE, LATENT_DIM,FEMALE_DATA_PATH
from utils import train_model, test_model, EarlyStopping, plot_confusion_matrix

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('--modelstr', default='resnet18', help='Model architecture to use')
@click.option('--experiment', default=1, type=int, help='Experiment number')
@click.option('--target_class', default='chimpanzee_ir', help='Target class for classification')
@click.option('--model_type', default='classifier', type=click.Choice(['classifier', 'contrastive']), help='Type of model to use')
def main(modelstr, experiment, target_class, model_type):
    """
    Main function to extract features from audio samples using a pre-trained model.
    """
    print(f"Starting feature extraction with the following parameters:")
    print(f"Model: {modelstr}")
    print(f"Experiment: {experiment}")
    print(f"Target class: {target_class}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
   

    dataset = SpectrogramDataset(DATA_PATH, duration=2, target_sample_rate=SAMPLING_RATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



    # Load the appropriate model
    if model_type == "classifier":
        print("Loading classifier model...")
        model = CustomCNNModel(num_classes=11, weights=None, modelstr=modelstr)
        model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_{modelstr}_{target_class}_experiment_{experiment}.pth', map_location=device))
        # model = torch.nn.Sequential(*list(model.base_model.children())[:-1], torch.nn.Flatten())
    elif model_type == "contrastive":
        print("Loading contrastive model...")
        features_model = ContrastiveCNN(latent_dim=LATENT_DIM, weights=None, modelstr=modelstr)
        print("We are here...")
        model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_type}_trained_model_{modelstr}_experiment_{experiment}.pt', map_location=device))

    model.to(device)
    model.eval()

    sample,label = next(iter(dataloader))

    # create unique tensor that contain 1 sample for each class 
    # to calculate the class activation map
    unique_sample_per_class = []
    unique_label_per_class = []
    for i in range(len(dataset.classes)):
        idx = np.where(label == i)[0][0]
        unique_sample_per_class.append(sample['data'][idx])
        unique_label_per_class.append(label[idx])

    # convert to torch tensors
    unique_sample_per_class = torch.stack(unique_sample_per_class)
    unique_label_per_class = torch.stack(unique_label_per_class)

    # print shape of the tensors
    # print(unique_sample_per_class.shape)
    # print(unique_label_per_class.shape)


    # create the ScoreCAM object
    if modelstr == 'resnet18':
        # cam = ScoreCAM(model, target_layer=model.base_model.layer4, batch_size = len(dataset.classes), input_shape = unique_sample_per_class.shape[1:])
        cam = ScoreCAM(model, target_layer=model.base_model.layer4)
    elif modelstr == 'dense121':
        # cam = ScoreCAM(model, target_layer=model.base_model.features.denseblock4.denselayer16, batch_size = len(dataset.classes), input_shape = unique_sample_per_class.shape[1:])
        cam = ScoreCAM(model, target_layer=model.base_model.features.denseblock4.denselayer16)
    
    with torch.no_grad():
        out = model(unique_sample_per_class[0].unsqueeze(0).to(device))
        cam_out = cam(class_idx=unique_label_per_class[0].item())
        print(out.shape, cam_out[0].shape)


    # Normalize CAM output
    # cam_normalized = (cam_out[0].squeeze().numpy() - cam_out[0].squeeze().numpy().min()) / (cam_out[0].squeeze().numpy().max() - cam_out[0].squeeze().numpy().min())



    result = overlay_mask(unique_sample_per_class[0], to_pil_image(cam_out[0], mode="F"), alpha=0.5)

    plt.figure(figsize=(10, 5))
    plt.imshow(unique_sample_per_class[0].squeeze().numpy(), cmap='gray', aspect='auto')
    plt.imshow(cam_out[0].squeeze().numpy(), cmap='plasma', alpha=0.5, aspect='auto')
    # plt.colorbar(label='CAM intensity')

    predicted_class = torch.argmax(out, dim=1).item()
    plt.title(f'Spectrogram with CAM overlay (Predicted Class: {predicted_class})')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()