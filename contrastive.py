from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import MAudioDataset,SpectrogramDataset
from models import ContrastiveCNN
from config import BATCH_SIZE, PT_LEARNING_RATE, EPOCHS, SEED, MODELS_PATH, RESULTS_PATH,MODELSTRS,SAMPLING_RATE, LATENT_DIM,HUMAN_DATA_PATH,DATA_PATH,CHIMPANZEE_DATA_PATH
from utils import train_model, test_model, EarlyStopping, plot_confusion_matrix
from losses import BatchAllTtripletLoss,SupConLoss
from tqdm import tqdm  # Add tqdm for progress bar
import click

# Set random seed for reproducibility
# torch.manual_seed(SEED)
# np.random.seed(SEED)

modelstr = MODELSTRS[0] #'resnet18

experiment = 1

target_class = 'chimpanzee_ir'
# target_class = 'human_ir'
contrastive_method = 'triplet'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model, trainloader, criterion, optimizer, device, num_epochs=10):
    # Initialize a list to store training losses
    train_losses = []
    # Training loop
    
    # Set model to training mode
    model.train()
    print("model.train mode initialized")
    # print("We are using device", device )
    for epoch in range(num_epochs):

        # Initialize variables to track training loss for this epoch
        running_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):

            # print("batch_idx", batch_idx)

            inputs = inputs['data'].to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)#.to(torch.long)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_train_loss += loss.item()
            num_train_batches += 1
  
            # Print progress message every, say, 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(trainloader)}] - "
                      f"Train Loss: {loss.item():.4f}")

        # Calculate the average training loss for this epoch
        epoch_train_loss = running_train_loss / num_train_batches
        train_losses.append(epoch_train_loss)
        torch.save(model.state_dict(),  f'{MODELS_PATH}/contrastive_model_{modelstr}_{target_class}_{contrastive_method}_experiment_{experiment}.pt')

        # Print the training loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}")
        # scheduler.step()
    print("Training completed.")

    return model, train_losses

@click.command()
@click.option('--experiment', type=int, default=1, help='Experiment number')
@click.option('--target_class', type=click.Choice(['chimpanzee_ir', 'human_ir']), default='chimpanzee_ir', help='Target class')
@click.option('--modelstr', type=click.Choice(MODELSTRS), default=MODELSTRS[0], help='Model string')
@click.option('--contrastive_method', type=click.Choice(['triplet', 'supcon']), default='triplet', help='Contrastive method')
def main(experiment, target_class, modelstr, contrastive_method):
    print(f"Running experiment {experiment} with target class {target_class}, model {modelstr}, and contrastive method {contrastive_method}")

    if target_class == 'chimpanzee_ir':
        dataset = SpectrogramDataset(f'{CHIMPANZEE_DATA_PATH}/train', duration=2, target_sample_rate=SAMPLING_RATE)
    else:
        dataset = SpectrogramDataset(HUMAN_DATA_PATH, duration=2, target_sample_rate=SAMPLING_RATE)

    print("Dataset size", len(dataset))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize the model
    model = ContrastiveCNN(latent_dim=LATENT_DIM, weights=None, modelstr=modelstr).to(device)

    if contrastive_method == 'triplet':
        criterion = BatchAllTtripletLoss()
    else:
        criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=PT_LEARNING_RATE)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    model, train_losses = train_loop(model, dataloader, criterion, optimizer, device, num_epochs=EPOCHS)

    # Save the model 
    torch.save(model.state_dict(), f'{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_experiment_{experiment}.pt')

    # Create a dictionary to store the collected results
    results_dict = {
        'train_losses': train_losses
    }

    # Create a pandas DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Define the file path to write the DataFrame
    output_file = f'{RESULTS_PATH}/contrastive_{modelstr}_{target_class}_{contrastive_method}_training_results_{experiment}.csv'

    # Write the DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)

    print(f'Results saved to {output_file}')
    print('Done', train_losses)

if __name__ == "__main__":
    main()