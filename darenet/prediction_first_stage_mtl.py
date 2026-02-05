"""
Multi-Task Learning Prediction Script
Evaluates both age estimation and Alzheimer's classification
"""
import os
import torch
import numpy as np
import torch.nn as nn
from utils.config import opt
from model import ACDense
from load_data import IMG_Folder, Integer_Multiple_Batch_Size
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def test_mtl(test_loader, model, device, save_results=True):
    """
    Test MTL model on both age estimation and diagnosis classification tasks
    
    Returns:
        age_mae: Mean Absolute Error for age
        age_rmse: Root Mean Squared Error for age
        age_corr: Correlation coefficients (Pearson, Spearman)
        dx_acc: Classification accuracy
        dx_report: Classification report (precision, recall, F1)
        results_dict: Dictionary with all predictions and targets
    """
    model.eval()
    
    all_age_pred = []
    all_age_true = []
    all_dx_pred = []
    all_dx_true = []
    all_sids = []
    all_gender = []
    
    print("========= Testing MTL Model =========")
    
    with torch.no_grad():
        for i, (img, sid, age_target, gender, dx_target) in enumerate(test_loader):
            # Prepare inputs
            img = img.to(device)
            age_target_np = age_target.numpy()
            dx_target_np = dx_target.numpy()
            
            # Convert gender to one-hot if needed
            if opt.use_gender:
                gender = torch.unsqueeze(gender, 1)
                gender = torch.zeros(gender.shape[0], 2).scatter_(1, gender, 1)
                gender = gender.to(device).type(torch.FloatTensor)
            
            # Forward pass
            age_pred, cls_logits = model(img, gender)
            
            # Get predictions
            age_pred_np = age_pred.cpu().numpy().flatten()
            dx_pred_np = cls_logits.argmax(1).cpu().numpy()
            
            # Store results
            all_age_pred.extend(age_pred_np)
            all_age_true.extend(age_target_np)
            all_dx_pred.extend(dx_pred_np)
            all_dx_true.extend(dx_target_np)
            all_sids.extend(sid)
            all_gender.extend(gender.cpu().numpy()[:, 1] if opt.use_gender else [0]*len(sid))
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_loader)} batches")
    
    # Convert to numpy arrays
    all_age_pred = np.array(all_age_pred)
    all_age_true = np.array(all_age_true)
    all_dx_pred = np.array(all_dx_pred)
    all_dx_true = np.array(all_dx_true)
    
    # =========== Age Estimation Metrics =========== #
    age_mae = mean_absolute_error(all_age_true, all_age_pred)
    age_rmse = np.sqrt(mean_squared_error(all_age_true, all_age_pred))
    pearson_r, pearson_p = pearsonr(all_age_true, all_age_pred)
    spearman_r, spearman_p = spearmanr(all_age_true, all_age_pred)
    
    # Brain Age Gap (BAG)
    bag = all_age_pred - all_age_true
    
    print("\n========= Age Estimation Results =========")
    print(f"MAE:  {age_mae:.3f} years")
    print(f"RMSE: {age_rmse:.3f} years")
    print(f"Pearson  R: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman R: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"BAG Mean: {bag.mean():.3f} ± {bag.std():.3f} years")
    
    # =========== Classification Metrics =========== #
    dx_acc = (all_dx_pred == all_dx_true).mean()
    dx_labels = ['Norm', 'MCI', 'AD']
    dx_class_labels = [0, 1, 2]  # Numeric labels for sklearn functions
    
    print("\n========= Alzheimer's Classification Results =========")
    print(f"Overall Accuracy: {dx_acc:.4f}")
    
    # Check unique classes in predictions and true labels
    unique_pred = np.unique(all_dx_pred)
    unique_true = np.unique(all_dx_true)
    print(f"Unique predicted classes: {unique_pred}")
    print(f"Unique true classes: {unique_true}")
    
    print("\nClassification Report:")
    # Use labels parameter to ensure all classes are included even if not present
    print(classification_report(all_dx_true, all_dx_pred, labels=dx_class_labels, target_names=dx_labels, digits=4))
    
    print("\nConfusion Matrix:")
    # Use labels parameter for confusion matrix as well
    cm = confusion_matrix(all_dx_true, all_dx_pred, labels=dx_class_labels)
    print(cm)
    
    # =========== Brain Age Gap by Diagnosis =========== #
    print("\n========= Brain Age Gap by Diagnosis =========")
    for dx_idx, dx_name in enumerate(dx_labels):
        mask = (all_dx_true == dx_idx)
        if mask.sum() > 0:
            bag_dx = bag[mask]
            print(f"{dx_name:6s}: {bag_dx.mean():6.2f} ± {bag_dx.std():.2f} years (n={mask.sum()})")
    
    # =========== Save Results =========== #
    if save_results:
        results_dict = {
            'sid': all_sids,
            'age_true': all_age_true,
            'age_pred': all_age_pred,
            'age_gap': bag,
            'dx_true': all_dx_true,
            'dx_pred': all_dx_pred,
            'gender': all_gender
        }
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        df['dx_true_name'] = df['dx_true'].map({0: 'Norm', 1: 'MCI', 2: 'AD'})
        df['dx_pred_name'] = df['dx_pred'].map({0: 'Norm', 1: 'MCI', 2: 'AD'})
        csv_path = os.path.join(opt.output_dir, 'mtl_test_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save numpy arrays
        npz_path = os.path.join(opt.output_dir, 'mtl_test_results.npz')
        np.savez(npz_path, **results_dict)
        print(f"Results saved to: {npz_path}")
        
        # =========== Generate Plots =========== #
        plot_results(all_age_true, all_age_pred, bag, all_dx_true, dx_labels, opt.output_dir)
    
    return {
        'age_mae': age_mae,
        'age_rmse': age_rmse,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'dx_acc': dx_acc,
        'bag_mean': bag.mean(),
        'bag_std': bag.std()
    }


def plot_results(age_true, age_pred, bag, dx_true, dx_labels, output_dir):
    """Generate visualization plots for MTL results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Age Prediction Scatter Plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(age_true, age_pred, alpha=0.5, s=20)
    ax1.plot([age_true.min(), age_true.max()], [age_true.min(), age_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True Age (years)', fontsize=12)
    ax1.set_ylabel('Predicted Age (years)', fontsize=12)
    ax1.set_title('Age Prediction', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    mae = mean_absolute_error(age_true, age_pred)
    r, _ = pearsonr(age_true, age_pred)
    ax1.text(0.05, 0.95, f'MAE: {mae:.2f}\nR: {r:.3f}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Age Prediction by Diagnosis
    ax2 = plt.subplot(2, 3, 2)
    colors = ['green', 'orange', 'red']
    for dx_idx, (dx_name, color) in enumerate(zip(dx_labels, colors)):
        mask = (dx_true == dx_idx)
        ax2.scatter(age_true[mask], age_pred[mask], alpha=0.6, s=20, label=dx_name, c=color)
    ax2.plot([age_true.min(), age_true.max()], [age_true.min(), age_true.max()], 'k--', lw=2)
    ax2.set_xlabel('True Age (years)', fontsize=12)
    ax2.set_ylabel('Predicted Age (years)', fontsize=12)
    ax2.set_title('Age Prediction by Diagnosis', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Brain Age Gap Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(bag, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', lw=2)
    ax3.set_xlabel('Brain Age Gap (years)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Brain Age Gap Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Brain Age Gap by Diagnosis (Box Plot)
    ax4 = plt.subplot(2, 3, 4)
    bag_by_dx = [bag[dx_true == i] for i in range(len(dx_labels))]
    bp = ax4.boxplot(bag_by_dx, labels=dx_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.axhline(y=0, color='red', linestyle='--', lw=2)
    ax4.set_xlabel('Diagnosis', fontsize=12)
    ax4.set_ylabel('Brain Age Gap (years)', fontsize=12)
    ax4.set_title('Brain Age Gap by Diagnosis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    dx_class_labels = [0, 1, 2]  # Numeric labels for sklearn functions
    cm = confusion_matrix(dx_true, np.digitize(bag, bins=[-100, -5, 5, 100])-1, labels=dx_class_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=dx_labels, yticklabels=dx_labels, ax=ax5, cbar_kws={'label': 'Proportion'})
    ax5.set_xlabel('Predicted Diagnosis', fontsize=12)
    ax5.set_ylabel('True Diagnosis', fontsize=12)
    ax5.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 6. Age Residuals
    ax6 = plt.subplot(2, 3, 6)
    residuals = age_pred - age_true
    ax6.scatter(age_true, residuals, alpha=0.5, s=20)
    ax6.axhline(y=0, color='red', linestyle='--', lw=2)
    ax6.set_xlabel('True Age (years)', fontsize=12)
    ax6.set_ylabel('Residuals (Pred - True)', fontsize=12)
    ax6.set_title('Age Prediction Residuals', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'mtl_evaluation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()


def main():
    """Main function for MTL model evaluation"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("========= Loading MTL Model =========")
    print(f"Model: {opt.model}")
    print(f"Output Dir: {opt.output_dir}")
    
    # Load data
    test_data = Integer_Multiple_Batch_Size(
        IMG_Folder(opt.excel_path, opt.test_folder, use_diagnosis=True),
        opt.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Build model
    if opt.model == 'DARENet' or opt.model == 'ScaleDenseMTL':
        model = ACDense.DARENet(8, 5, opt.use_gender, num_classes=3)
    else:
        raise ValueError(f"This script is for MTL models. Got: {opt.model}")
    
    model = nn.DataParallel(model).to(device)
    
    # Load checkpoint
    model_path = os.path.join(opt.output_dir, opt.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # PyTorch 2.6+ compatibility: set weights_only=False for checkpoint loading
    try:
        checkpoint = torch.load(model_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Run evaluation
    metrics = test_mtl(test_loader, model, device, save_results=True)
    
    print("\n========= Summary =========")
    print(f"Age MAE:  {metrics['age_mae']:.3f} years")
    print(f"Age RMSE: {metrics['age_rmse']:.3f} years")
    print(f"Pearson R: {metrics['pearson_r']:.4f}")
    print(f"Diagnosis Accuracy: {metrics['dx_acc']:.4f}")
    print(f"Brain Age Gap: {metrics['bag_mean']:.2f} ± {metrics['bag_std']:.2f} years")
    
    return metrics


if __name__ == '__main__':
    main()

