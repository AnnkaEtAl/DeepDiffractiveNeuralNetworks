import numpy as np 
import pandas as pd
import os, time, sys
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
import importlib
import matplotlib.pyplot as plt
from skimage import measure
import os, json
import math

def get_detector_params(params, best_params):
    Dmeta=params["Nmeta"]*params["pixelsize"]
    D=Dmeta/best_params["magIn"]
    DDet=D*best_params['magOut']
    x = y = np.linspace(-D/2, D/2, params["N"])
    file= os.path.join(os.getcwd(),r'detector_masks',"10_classes_standard.npy")
    
    return DDet, x, y, file


#plotting weights and training history
def plot_model_parameters(weights, history):
    from matplotlib.ticker import MaxNLocator

    plt.figure(figsize=(10, 4))
    cmap = 'viridis'  # Good for phase-like visualization
    for i, w in enumerate(weights):
        ax = plt.subplot(1, len(weights), i + 1)
        
        
        ax.axis('off')
        im = ax.imshow(w*2*np.pi, cmap=cmap, vmin=0, vmax=2*np.pi) if i %2== 0 else ax.imshow(w, cmap=cmap)
        
        # Colorbar matching the subplot height
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal')
        
        if i %2 == 0:
            im.set_clim(0, 2*np.pi)  # Explicitly set for phase
            ax.set_title(f"Phase Mask Layer")
            ticks = [0, np.pi, 2*np.pi]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
            
        else:
            ax.set_title("Amplitude")

    plt.tight_layout()
    plt.show()


    train_keys = [key for key in history if not key.startswith('val_') and key!="lr"]
    num_plots = len(train_keys)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten to 1D for easy indexing

    for idx, key in enumerate(train_keys):
        ax = axes[idx]
        val_key = 'val_' + key

        ax.plot(history[key], label=f'Train {key}')
        if val_key in history:
            ax.plot(history[val_key], label=f'Val {key}')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(f'{key} over Epochs')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(train_keys), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_optimization(optimizer_res):
    penalty_exist="penalty"  in optimizer_res[0]["params"].keys()

    targets = [r['target'] for r in optimizer_res]
    magIns = [r['params']['magIn'] for r in optimizer_res]
    magOuts = [r['params']['magOut'] for r in optimizer_res]
    if penalty_exist:
        penalties=[r['params']['penalty'] for r in optimizer_res]
    iterations = list(range(len(optimizer_res)))

    # Compute cumulative (best-so-far) accuracy
    cumulative_targets = np.maximum.accumulate(targets)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top plot: Cumulative Target vs Iteration
    ax1.plot(iterations, cumulative_targets, label='Cumulative Best Target', color='purple', marker='o')
    ax1.scatter(iterations, targets, label='Target',color='purple', marker='o')

    ax1.set_ylabel('Cumulative Accuracy')
    ax1.grid(True)
    ax1.legend()

    # Bottom plot: Parameters vs Iteration
    ax2.scatter(iterations, magIns, label='magIn', color='blue', alpha=0.7)
    ax2.scatter(iterations, magOuts, label='magOut', color='green', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MagIn/MagOut Value', color="blue")
    ax2.grid(True)
    ax2.legend()
    if penalty_exist:
        ax3 = ax2.twinx()  # Create second y-axis that shares the same x-axis
        ax3.scatter(iterations, penalties, label='penalty', color='red', alpha=0.7)
        ax3.set_ylabel('Penalty', color='red')

    plt.tight_layout()
    plt.show()




def get_example_indices(model, x_data, y_true, num_examples=3):
    """
    Returns indices of correctly and incorrectly classified samples.
    
    Args:
        model: Trained TensorFlow/Keras model.
        x_data: Input images (normalized and correctly shaped).
        y_true: True labels.
        num_examples: Number of correct and incorrect indices to return.
        
    Returns:
        correct_indices: List of indices of correctly predicted samples.
        incorrect_indices: List of indices of incorrectly predicted samples.
    """
    y_pred = model(x_data, training=False)
    y_pred_classes = np.argmax(y_pred, axis=1)
    

    correct = np.where(y_pred_classes == y_true)[0]
    incorrect = np.where(y_pred_classes != y_true)[0]
    
    correct_indices = correct[:num_examples].tolist()
    incorrect_indices = incorrect[:num_examples].tolist()
    
    return correct_indices, incorrect_indices

def get_zoom(detector, margin=20):
    # Get combined bounding box of all detectors
    combined_mask = np.any(detector == np.max(detector[0]), axis=0)
    img_size=np.shape(detector)[1]
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Distances from bbox edges to image edges
    dist_top = row_min
    dist_bottom = img_size - 1 - row_max
    dist_left = col_min
    dist_right = img_size - 1 - col_max

    # Max distance from any side — that's minimal zoom needed
    zoom = max(dist_top, dist_bottom, dist_left, dist_right) - margin

    return zoom


def plotFields(model, testImages, labels_testImages, 
               prediction_testImages, planes, detector, zoom, bigger_detector=None):
    layers = model.layers
    outputs = []

    # Get field outputs at specified planes
    for layer_index in planes:
        intermediate_model = Model(inputs=model.input, outputs=layers[layer_index].output)
        field = np.abs(intermediate_model.predict(testImages, verbose=False)) ** 2
        outputs.append(field)

    # Compute detector-masked output and apply zoom
    #masked = outputs[-1] * np.sum(detector, axis=0)
    #outputs.append(masked[:, zoom:-zoom, zoom:-zoom])

    #NOT MASKED
    outputs.append(outputs[-1][:, zoom:-zoom, zoom:-zoom])


    num_samples = len(testImages)
    num_outputs = len(outputs)
    fig, axes = plt.subplots(num_samples, num_outputs, figsize=(2*num_outputs, 2 * num_samples), constrained_layout=True)

    # Precompute contours for all 10 detectors
    all_detector_contours = []
    for k in range(len(detector)):
        mask = detector[k, zoom:-zoom, zoom:-zoom]
        contours = measure.find_contours(mask, np.max(detector[k])/2)
        all_detector_contours.append(contours)

    if bigger_detector is not None:
        all_bigger_detector_contours = []
        for i in range(bigger_detector.shape[0]):
            # Find contours for each class patch at level 0.5 (to detect boundary between 0 and 1)
            mask2=bigger_detector[i, zoom:-zoom, zoom:-zoom]
            bigger_contours = measure.find_contours(mask2, np.max(bigger_detector[i])/2)
            all_bigger_detector_contours.append(bigger_contours)

    for j in range(num_samples):  # For each test image
        true_label = labels_testImages[j]
        pred_label = prediction_testImages[j]
        for i in range(num_outputs):  # For each output layer
            ax = axes[j, i] if num_samples > 1 else axes[i]
            im = ax.imshow(outputs[i][j], cmap='viridis', vmax=np.max(outputs[i][j]))
            ax.axis("off")
            # If it's the last column, plot all detector outlines
            if i == num_outputs - 1:
                for  label, contour_set in zip(range(0, len(detector)), all_detector_contours):
                    for contour in contour_set:
                        color='orange' #default color
                        if label == true_label and label == pred_label:
                            color = 'green'  # correct prediction, highlight green
                        elif label == true_label:
                            color = 'green'  # true label contour
                        elif label == pred_label:
                            color = 'red'  # wrong predicted contour

                        ax.plot(contour[:, 1], contour[:, 0], color='orange', linewidth=1.0, alpha=0.5)
                        x, y = contour[0, 1], contour[0, 0]  
                        ax.text(x+2, y-5, str(label), color=color, fontsize=8, weight='bold')
                if bigger_detector is not None:
                    for contour_set in all_bigger_detector_contours:
                        for contour in contour_set:
                            ax.plot(contour[:, 1], contour[:, 0], color='yellow', linewidth=1.0, alpha=0.4)
                #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    plt.show()



def save_model_metadata_csv(model_name:str,
                            model_params: dict,
                            metrics: dict,
                            filename: str='models_summary.csv',
                            result_path: str=None):
    """
    Save metadata to CSV file named `filename` inside `result_path`.
    If result_path is None, saves in current working directory.
    """

    # Determine full path for the CSV file
    if result_path:
        os.makedirs(result_path, exist_ok=True)  # make sure directory exists
        summary_csv_path = os.path.join(result_path, filename)
    else:
        summary_csv_path = filename  # current working directory

    row = {"model_name": model_name}
    # Flatten model params
    for key, val in model_params.items():
        row[key] = val

    # Add metrics
    for key, val in metrics.items():
        row[key] = val

    # Load existing or create new
    if os.path.exists(summary_csv_path):
        df = pd.read_csv(summary_csv_path)
        if model_name in df["model_name"].values:
            print(f"[!] Model '{model_name}' already in summary. Skipping.")
            return
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_csv_path, index=False)
    print(f"[+] Saved metadata for model '{model_name}'")