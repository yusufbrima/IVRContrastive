import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import librosa
import librosa.display
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import torchaudio.transforms as transforms
from dataset import SpectrogramDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.lines as mlines
from models import CustomCNNModel,FinetuningClassifier,ContrastiveCNN
from utils import AudioDataProcessor,sample_balanced_representations,compute_cosine_similarity,plot_similarity_heatmap
from config import DATA_PATH, BATCH_SIZE, MODELS_PATH, RESULTS_PATH,MODELSTRS,FIG_PATH,SAMPLING_RATE,FONTSIZE,LATENT_DIM,EXPORT_DATA_PATH


modelstrs = MODELSTRS
methods =  ['supcon','triplet']
target_class = "chimpanzee_ir"
model_type = "contrastive"
experiment = 1
females = "_females"
history = {'method':[], "score":[], "threshold":[]}
if __name__ == "__main__":
    print("Starting...")
    plt.figure(figsize=(10, 8))
    line_styles = ['-', '--', '-.', ':']
    model_labels = ["ResNet18 SupCon", "ResNet18 Triplet", "DenseNet121 SupCon", "DenseNet121 Triplet"]
    markers = ['p', 's', '^', 'D']  # Pentagon, Square, Triangle_up, Diamond
    color_list = ['y', 'g', 'r', 'm']
    # color_list_2 = ['b', 'c', 'k', 'orange']
    # Plotting data and storing handles
    handles = []
    idx = 0
    for  modelstr in modelstrs:
        for method in methods:
            print(f"Model: {modelstr}, Method: {method}")
            if females:
                data = np.load(f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}__{method}_experiment_{experiment}{females}.npz', allow_pickle=True)
            else:
                data = np.load(f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_{method}_experiment_{experiment}.npz', allow_pickle=True)
            features = data['features']
            labels = data['labels']
            label_names = data['label_classes']

            # print(f"Features shape: {features.shape} Labels shape: {labels.shape} modelstr: {modelstr} method: {method}")

            # Assuming 'embeddings' is your numpy array of shape (n_samples, latent_dim)
            similarity_matrix = cosine_similarity(features)

            # Now compute similarity scores for genuine and impostor pairs
            n_samples = len(features)
            genuine_scores = []
            impostor_scores = []

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j]:
                        genuine_scores.append(similarity_matrix[i, j])
                    else:
                        impostor_scores.append(similarity_matrix[i, j])

            genuine_scores = np.array(genuine_scores)
            impostor_scores = np.array(impostor_scores)

            # Compute the EER
            # Create labels for the scores
            scores = np.concatenate([genuine_scores, impostor_scores])
            labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))

            # Compute the ROC curve
            fpr, tpr, thresholds = roc_curve(labels, scores)

            # Interpolate the FPR and FNR
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer_threshold = interp1d(fpr, thresholds)(eer)

            print("EER: {:.2f}%".format(eer))
            print("Threshold at EER: {:.4f}".format(eer_threshold))

            history['method'].append(f"{modelstr}_{method}")
            history['score'].append(eer)
            history['threshold'].append(eer_threshold)

            fnr = 1 - tpr

            # Convert rates to percentages
            fpr100 = fpr * 100
            fnr100 = fnr * 100

            # Calculate the EER and the threshold
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer_threshold = interp1d(fpr, thresholds)(eer)
            eer_fpr = interp1d(fpr, fpr)(eer) * 100
            eer_fnr = interp1d(fnr, fnr)(eer) * 100


            # Plotting the DET curve
            lines, = plt.plot(fpr100, fnr100, label=f'{model_labels[idx]}', linewidth=2, linestyle=line_styles[idx], color=color_list[idx])

            # Marking the EER point
            plt.scatter([eer_fpr], [eer_fnr], marker=markers[idx], color=color_list[idx])  # EER point
            plt.text(eer_fpr, eer_fnr,f'', fontsize=FONTSIZE, color=color_list[idx], ha='right', va='bottom')
            
            # Create a custom handle with the line style and marker
            handle = mlines.Line2D([], [],color=color_list[idx], marker=markers[idx], linestyle=line_styles[idx], label=f'{model_labels[idx]}')
            handles.append(handle)
            idx += 1

            print("EER after: {:.2f}%".format(eer))
            print("Threshold at EER after: {:.4f}".format(eer_threshold))

            # Logarithmic scale for better visualization
            # plt.yscale('log')
            # plt.xscale('log')
    plt.xlabel('False Positive Rate (%)', fontsize=FONTSIZE)
    plt.ylabel('False Negative Rate (%)', fontsize=FONTSIZE)
    plt.title('Detection Error Tradeoff (DET) Curve with EER', fontsize=FONTSIZE)
    plt.grid(True)
    plt.minorticks_on()
    plt.legend(handles=handles, loc='best', fontsize=FONTSIZE)
    if females:
        plt.savefig(f'{FIG_PATH}/eer_scores_contrastive_experiment_{experiment}_{females}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{FIG_PATH}/eer_scores_contrastive_experiment_{experiment}.png', dpi=300, bbox_inches='tight')
    plt.show()



    df = pd.DataFrame(history)
    # Save the DataFrame
    if females:
        df.to_csv(f'{RESULTS_PATH}/eer_scores_contrastive_experiment_{experiment}_{females}.csv', index=False)
    else:
        df.to_csv(f'{RESULTS_PATH}/eer_scores_contrastive_experiment_{experiment}.csv', index=False)









