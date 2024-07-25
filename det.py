import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from config import  RESULTS_PATH, MODELSTRS, FIG_PATH, SAMPLING_RATE, FONTSIZE, LATENT_DIM, EXPORT_DATA_PATH

def load_data(file_path):
    """Load features, labels, and label classes from a .npz file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        return data['features'], data['labels'], data['label_classes']
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None, None

def compute_similarity(features):
    """Compute the cosine similarity matrix for the given features."""
    return cosine_similarity(features)

def compute_scores(similarity_matrix, labels):
    """Compute genuine and impostor scores from the similarity matrix and labels."""
    n_samples = len(labels)
    genuine_scores = []
    impostor_scores = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if labels[i] == labels[j]:
                genuine_scores.append(similarity_matrix[i, j])
            else:
                impostor_scores.append(similarity_matrix[i, j])

    return np.array(genuine_scores), np.array(impostor_scores)

def calculate_eer(genuine_scores, impostor_scores):
    """Calculate the Equal Error Rate (EER) and corresponding threshold."""
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return eer, eer_threshold, fpr, tpr

def plot_det_curve(fpr, tpr, eer, eer_threshold, model_labels, line_styles, color_list, markers, idx, handles):
    """Plot the DET curve and mark the EER point."""
    fnr = 1 - tpr
    fpr100 = fpr * 100
    fnr100 = fnr * 100

    # Plot the DET curve
    plt.plot(fpr100, fnr100, label=model_labels[idx], linewidth=2, linestyle=line_styles[idx], color=color_list[idx])

    # Mark the EER point
    eer_fpr = eer * 100
    eer_fnr = (1 - eer) * 100
    plt.scatter([eer_fpr], [eer_fnr], marker=markers[idx], color=color_list[idx])
    plt.text(eer_fpr, eer_fnr, f'{eer:.2f}', fontsize=FONTSIZE, color=color_list[idx], ha='right', va='bottom')

    # Create a custom handle with the line style and marker
    handle = mlines.Line2D([], [], color=color_list[idx], marker=markers[idx], linestyle=line_styles[idx], label=model_labels[idx])
    handles.append(handle)

def main():
    modelstrs = MODELSTRS
    methods = ['supcon', 'triplet']
    target_class = "chimpanzee_ir"
    model_type = "contrastive"
    experiment = 1
    females = ""
    history = {'method': [], "score": [], "threshold": []}
    
    plt.figure(figsize=(10, 8))
    line_styles = ['-', '--', '-.', ':']
    model_labels = ["ResNet18 SupCon", "ResNet18 Triplet", "DenseNet121 SupCon", "DenseNet121 Triplet"]
    markers = ['p', 's', '^', 'D']
    color_list = ['y', 'g', 'r', 'm']
    handles = []

    idx = 0
    for modelstr in modelstrs:
        for method in methods:
            print(f"Model: {modelstr}, Method: {method}")
            if females:
                file_path = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}__{method}_experiment_{experiment}{females}.npz'
            else:
                file_path = f'{EXPORT_DATA_PATH}/features_{model_type}_{modelstr}_{target_class}_{method}_experiment_{experiment}.npz'

            features, labels, label_names = load_data(file_path)
            if features is None or labels is None:
                continue

            print(f"Features shape: {features.shape} Labels shape: {labels.shape} modelstr: {modelstr} method: {method}")

            similarity_matrix = compute_similarity(features)
            genuine_scores, impostor_scores = compute_scores(similarity_matrix, labels)
            eer, eer_threshold, fpr, tpr = calculate_eer(genuine_scores, impostor_scores)

            print(f"EER: {eer:.2f}%")
            print(f"Threshold at EER: {eer_threshold:.4f}")

            history['method'].append(f"{modelstr}_{method}")
            history['score'].append(eer)
            history['threshold'].append(eer_threshold)

            plot_det_curve(fpr, tpr, eer, eer_threshold, model_labels, line_styles, color_list, markers, idx, handles)
            idx += 1

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
    if females:
        df.to_csv(f'{RESULTS_PATH}/eer_scores_contrastive_experiment_{experiment}_{females}.csv', index=False)
    else:
        df.to_csv(f'{RESULTS_PATH}/eer_scores_contrastive_experiment_{experiment}.csv', index=False)

if __name__ == "__main__":
    main()

