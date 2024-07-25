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
# from torchcam.methods import ScoreCAM,SmoothGradCAMpp
from camlib import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
# from torchcam.utils import overlay_mask
# from camviz import visualize
from PIL import Image
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import MAudioDataset, SpectrogramDataset
from models import CustomCNNModel, ContrastiveCNN, FinetuningClassifier
from config import DATA_PATH, BATCH_SIZE, SEED, MODELS_PATH, MODELSTRS, SAMPLING_RATE,FIG_PATH, FONTSIZE, LATENT_DIM,CHIMPANZEE_DATA_PATH

# Set random seed for reproducibility
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('--experiment', default=1, type=int, help='Experiment number')
@click.option('--target_class', default='chimpanzee_ir', help='Target class for classification')
@click.option('--ft', default=False, type=bool, help='Fine-tune model')
@click.option('--contrastive_method', default='triplet', help='Contrastive method to use for the model')
@click.option('--duration', default=2, type=int, help='Duration of audio samples')
def main(experiment, target_class, ft,contrastive_method, duration):
    """
    Main function to extract features from audio samples using a pre-trained model.
    """

    dataset = SpectrogramDataset(f"{CHIMPANZEE_DATA_PATH}/val", duration=duration, target_sample_rate=SAMPLING_RATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)






    if ft == True:
        # dense_model = CustomCNNModel(num_classes=11, weights=None, modelstr='dense121')
        model = ContrastiveCNN(latent_dim=LATENT_DIM, weights=None,  modelstr='dense121')
        dense_model = FinetuningClassifier(model, num_classes=11).to(device)
        dense_model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_ft_dense121_{target_class}_{contrastive_method}_experiment_{experiment}.pth', map_location=device))
        
        model = ContrastiveCNN(latent_dim=LATENT_DIM, weights=None,  modelstr='resnet18')
        resnet_model = FinetuningClassifier(model, num_classes=11).to(device)
        resnet_model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_ft_resnet18_{target_class}_{contrastive_method}_experiment_{experiment}.pth', map_location=device))
        save_prefix = f'ft_{contrastive_method}_'
    else:
        dense_max = 34
        resnet_max = 12
        dense_model = CustomCNNModel(num_classes=11, weights=None, modelstr='dense121')
        dense_model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_dense121_{target_class}_experiment_{dense_max}.pth', map_location=device))

        resnet_model = CustomCNNModel(num_classes=11, weights=None, modelstr='resnet18')
        resnet_model.load_state_dict(torch.load(f'{MODELS_PATH}/custom_resnet18_{target_class}_experiment_{resnet_max}.pth', map_location=device))
        save_prefix = ''

    dense_model.to(device)
    dense_model.eval()

    resnet_model.to(device)
    resnet_model.eval()

    sample,label = next(iter(dataloader))

    # create unique tensor that contain 1 sample for each class 
    # to calculate the class activation map
    unique_sample_per_class = []
    unique_label_per_class = []
    flag =  11

    for data,labels in dataloader:
        sample, labels = data['data'], labels
        for i in range(len(labels)):
            label_value = int(labels[i].item())
            if label_value not in unique_label_per_class:
                unique_sample_per_class.append(sample[i])
                unique_label_per_class.append(label_value)
            if len(unique_label_per_class) == flag:
                break
        if len(unique_label_per_class) == flag:
            break

    # convert to torch tensors
    unique_sample_per_class = torch.stack(unique_sample_per_class)
    # unique_label_per_class = torch.stack(unique_label_per_class)

    method_list = ['scorecam'] #'gradcam', 'gradcampp', 
    modelstr_list = ['resnet18', 'dense121']
    for method in method_list:
        print(f"Processing method: {method}")
        cam_output_list = []
        class_label_list = []
        image_list = []
        for modelstr_item in modelstr_list:
            for i in range(len(unique_sample_per_class)):
                # create the ScoreCAM object
                if modelstr_item == 'resnet18':
                    if ft == True:
                        # write a print statement here
                        # print("We are here now, condition is true")
                        target_layer = resnet_model.feature_extractor.base_model.layer4[1].conv2
                    else:
                        target_layer = resnet_model.base_model.layer4[1].conv2
                    if method == 'gradcam':
                        wrapped_model = GradCAM(resnet_model, target_layer=target_layer)
                    elif method == 'gradcampp':
                        wrapped_model = GradCAMpp(resnet_model, target_layer=target_layer)
                    elif method == 'smoothgradcampp':
                        wrapped_model = SmoothGradCAMpp(resnet_model, target_layer=resnet_model.base_model.layer4[1].conv2,n_samples=25, stdev_spread=0.15)
                    elif method == 'scorecam':
                        wrapped_model = ScoreCAM(resnet_model, target_layer=target_layer)
                    # wrapped_model = GradCAM(resnet_model, target_layer=resnet_model.base_model.layer4[1].conv2)
                elif modelstr_item == 'dense121':
                    if ft == True:
                        target_layer = dense_model.feature_extractor.base_model.features.denseblock4.denselayer16.conv2
                    else:
                        target_layer = dense_model.base_model.features.denseblock4.denselayer16.conv2
                    if method == 'gradcam':
                        wrapped_model = GradCAM(dense_model, target_layer=target_layer)
                    elif method == 'gradcampp':
                        wrapped_model = GradCAMpp(dense_model, target_layer=target_layer)
                    elif method == 'smoothgradcampp':
                        wrapped_model = SmoothGradCAMpp(dense_model, target_layer=target_layer,n_samples=25, stdev_spread=0.15)
                    elif method == 'scorecam':
                        wrapped_model = ScoreCAM(dense_model, target_layer=target_layer)
                
                # with torch.inference_mode():
                cam, idx = wrapped_model(unique_sample_per_class[i].unsqueeze(0).to(device))
                cam_output_list.append(cam)
                class_label_list.append(f"GT: {dataset.classes[unique_label_per_class[i]]} - PT: {dataset.classes[idx]}")
                image_list.append(unique_sample_per_class[i])
        # print("We are here now")
        print(cam_output_list[0].shape)
        sentinel = 0
        fig, axs = plt.subplots(2, 11, figsize=(25, 6))
        for ax in axs.flatten():
            ax.imshow(image_list[sentinel].squeeze().cpu().numpy(), origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
            ax.imshow(cam_output_list[sentinel].squeeze().cpu().numpy(), cmap='magma', alpha=0.5, aspect='auto', origin='lower')
            ax.set_title(class_label_list[sentinel], fontsize=FONTSIZE)
            
            # Hide tick marks and labels, except for 10th and 11th plots
            ax.set_xticks([])
            if sentinel != 0 and sentinel != 11:
                ax.set_yticks([])

            if sentinel == 0:
                ax.set_ylabel('Dense121', fontsize=FONTSIZE)
            if sentinel == 11:
                ax.set_ylabel('Resnet18', fontsize=FONTSIZE)
            sentinel += 1
        plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust these values as needed
        plt.tight_layout()
        fig.savefig(f'{FIG_PATH}/librosa_{save_prefix}{method}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()