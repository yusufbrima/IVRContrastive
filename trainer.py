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
from dataset import SpectrogramDataset
from models import CustomCNNModel
from config import DATA_PATH, CLASSIFIER_BATCH_SIZE, LEARNING_RATE, SEED, MODELS_PATH, RESULTS_PATH,SAMPLING_RATE,FT_EPOCHS,CHIMPANZEE_DATA_PATH,CLASS_WEIGHTS
from utils import train_model, test_model, EarlyStopping, plot_confusion_matrix
from tqdm import tqdm  # Add tqdm for progress bar
import click

# Set random seed for reproducibility
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('--modelstr', default='dense121', help='Model architecture to use')
@click.option('--experiment', default=0, type=int, help='Experiment number')
@click.option('--target_class', default='chimpanzee_ir', help='Target class for classification')
def main(modelstr, experiment,target_class):
    # Load the dataset
    train_ds = SpectrogramDataset(f"{CHIMPANZEE_DATA_PATH}/train", duration=2, target_sample_rate=SAMPLING_RATE)

    test_dataset = SpectrogramDataset(f"{CHIMPANZEE_DATA_PATH}/val", duration=2, target_sample_rate=SAMPLING_RATE)

    # Define the sizes of the splits
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
 

    # Split the dataset
    train_dataset, val_dataset = random_split(train_ds, [train_size, val_size])

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=2)



    sample, label = next(iter(train_loader))

    input_shape = sample['data'].shape[1:]  

    num_classes = len(train_ds.classes)

    model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # print the length of the datasets 
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=FT_EPOCHS, device=device, save_path=f'bkp_saved_model_{modelstr}_{target_class}_experiment_{experiment}.pth')
    
    # # Save the trained model
    torch.save(model.state_dict(), f'{MODELS_PATH}/custom_{modelstr}_{target_class}_experiment_{experiment}.pth')

    # Print the training and validation loss and accuracy history
    print('Training and validation loss and accuracy history:')
    print(history)

    # # Example usage:
    test_loss, test_acc, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)

    # # Save the test labels and predictions
    test_results = {'labels': all_labels, 'preds': all_preds}
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_scores_{target_class}_experiment_{experiment}.csv', index=False)

    # # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{RESULTS_PATH}/{modelstr}_history_{target_class}_experiment_{experiment}.csv', index=False)

    # # Save the test results
    test_results = {'test_loss': test_loss, 'test_acc': test_acc}
    test_results_df = pd.DataFrame(test_results, index=[0])
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_results_{target_class}_experiment_{experiment}.csv', index=False)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__=="__main__":
    main()
