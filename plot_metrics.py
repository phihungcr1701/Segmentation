import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_metrics(model_name, metrics_path='metrics/'):
    """
    Plot training metrics from JSON file.
    
    Args:
        model_name: Name of the model (must match the saved metrics file)
        metrics_path: Path to the metrics directory
    """
    metrics_file = Path(metrics_path) / f'{model_name}_metrics.json'
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    epochs_data = data['epochs']
    
    if len(epochs_data) == 0:
        print("No metrics data found")
        return
    
    # Extract data for plotting
    epochs = [m['epoch'] for m in epochs_data]
    train_losses = [m['train_loss'] for m in epochs_data]
    val_losses = [m['val_loss'] for m in epochs_data]
    
    # Train metrics
    train_mean_dices = [m['train_metrics']['mean_dice'] for m in epochs_data]
    train_mean_ious = [m['train_metrics']['mean_iou'] for m in epochs_data]
    train_mean_accuracies = [m['train_metrics']['mean_accuracy'] for m in epochs_data]
    train_mean_precisions = [m['train_metrics']['mean_precision'] for m in epochs_data]
    train_mean_recalls = [m['train_metrics']['mean_recall'] for m in epochs_data]
    train_overall_pixel_accs = [m['train_metrics']['overall_pixel_acc'] for m in epochs_data]
    
    # Validation metrics
    val_mean_dices = [m['val_metrics']['mean_dice'] for m in epochs_data]
    val_mean_ious = [m['val_metrics']['mean_iou'] for m in epochs_data]
    val_mean_accuracies = [m['val_metrics']['mean_accuracy'] for m in epochs_data]
    val_mean_precisions = [m['val_metrics']['mean_precision'] for m in epochs_data]
    val_mean_recalls = [m['val_metrics']['mean_recall'] for m in epochs_data]
    val_overall_pixel_accs = [m['val_metrics']['overall_pixel_acc'] for m in epochs_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Training Metrics - {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Dice Score
    axes[0, 1].plot(epochs, train_mean_dices, label='Train Dice', marker='o', color='green')
    axes[0, 1].plot(epochs, val_mean_dices, label='Val Dice', marker='s', color='darkgreen')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Mean Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: IoU
    axes[1, 0].plot(epochs, train_mean_ious, label='Train IoU', marker='o', color='orange')
    axes[1, 0].plot(epochs, val_mean_ious, label='Val IoU', marker='s', color='darkorange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Mean IoU (Intersection over Union)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Accuracy Metrics
    axes[1, 1].plot(epochs, train_mean_accuracies, label='Train Accuracy', marker='o', color='blue')
    axes[1, 1].plot(epochs, val_mean_accuracies, label='Val Accuracy', marker='s', color='darkblue')
    axes[1, 1].plot(epochs, train_overall_pixel_accs, label='Train Pixel Acc', marker='^', color='purple')
    axes[1, 1].plot(epochs, val_overall_pixel_accs, label='Val Pixel Acc', marker='v', color='darkviolet')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Precision
    axes[2, 0].plot(epochs, train_mean_precisions, label='Train Precision', marker='o', color='red')
    axes[2, 0].plot(epochs, val_mean_precisions, label='Val Precision', marker='s', color='darkred')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].set_title('Mean Precision')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Recall
    axes[2, 1].plot(epochs, train_mean_recalls, label='Train Recall', marker='o', color='brown')
    axes[2, 1].plot(epochs, val_mean_recalls, label='Val Recall', marker='s', color='maroon')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Recall')
    axes[2, 1].set_title('Mean Recall')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(metrics_path) / f'{model_name}_metrics_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Display per-class metrics in a separate figure
    plot_per_class_metrics(epochs_data, model_name, metrics_path)
    
    plt.show()


def plot_per_class_metrics(epochs_data, model_name, metrics_path='metrics/'):
    """
    Plot per-class metrics (Dice, IoU, etc.)
    """
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    
    # Extract per-class data
    epochs = [m['epoch'] for m in epochs_data]
    train_dice_per_class = {i: [] for i in range(6)}
    val_dice_per_class = {i: [] for i in range(6)}
    train_iou_per_class = {i: [] for i in range(6)}
    val_iou_per_class = {i: [] for i in range(6)}
    train_precision_per_class = {i: [] for i in range(6)}
    val_precision_per_class = {i: [] for i in range(6)}
    train_recall_per_class = {i: [] for i in range(6)}
    val_recall_per_class = {i: [] for i in range(6)}
    
    for m in epochs_data:
        for i in range(6):
            train_dice_per_class[i].append(m['train_metrics']['dice_scores'][i])
            val_dice_per_class[i].append(m['val_metrics']['dice_scores'][i])
            train_iou_per_class[i].append(m['train_metrics']['ious'][i])
            val_iou_per_class[i].append(m['val_metrics']['ious'][i])
            train_precision_per_class[i].append(m['train_metrics']['precisions'][i])
            val_precision_per_class[i].append(m['val_metrics']['precisions'][i])
            train_recall_per_class[i].append(m['train_metrics']['recalls'][i])
            val_recall_per_class[i].append(m['val_metrics']['recalls'][i])
    
    # Create figure for per-class metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Per-Class Metrics - {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Dice per class
    for i in range(6):
        axes[0, 0].plot(epochs, train_dice_per_class[i], label=f'{class_names[i]} (Train)', marker='o', linestyle='-', alpha=0.7)
        axes[0, 0].plot(epochs, val_dice_per_class[i], label=f'{class_names[i]} (Val)', marker='s', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].set_title('Dice Score per Class (Train vs Val)')
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: IoU per class
    for i in range(6):
        axes[0, 1].plot(epochs, train_iou_per_class[i], label=f'{class_names[i]} (Train)', marker='o', linestyle='-', alpha=0.7)
        axes[0, 1].plot(epochs, val_iou_per_class[i], label=f'{class_names[i]} (Val)', marker='s', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('IoU per Class (Train vs Val)')
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision per class
    for i in range(6):
        axes[1, 0].plot(epochs, train_precision_per_class[i], label=f'{class_names[i]} (Train)', marker='o', linestyle='-', alpha=0.7)
        axes[1, 0].plot(epochs, val_precision_per_class[i], label=f'{class_names[i]} (Val)', marker='s', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision per Class (Train vs Val)')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Recall per class
    for i in range(6):
        axes[1, 1].plot(epochs, train_recall_per_class[i], label=f'{class_names[i]} (Train)', marker='o', linestyle='-', alpha=0.7)
        axes[1, 1].plot(epochs, val_recall_per_class[i], label=f'{class_names[i]} (Val)', marker='s', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall per Class (Train vs Val)')
    axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(metrics_path) / f'{model_name}_per_class_metrics_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-class plot saved to {output_path}")
    
    plt.show()


def print_metrics_summary(model_name, metrics_path='metrics/'):
    """
    Print summary of metrics from JSON file.
    """
    metrics_file = Path(metrics_path) / f'{model_name}_metrics.json'
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    epochs_data = data['epochs']
    
    if len(epochs_data) == 0:
        print("No metrics data found")
        return
    
    # Print summary for last epoch
    last_epoch = epochs_data[-1]
    print(f"\n{'='*80}")
    print(f"Metrics Summary - {model_name} (Last Epoch: {last_epoch['epoch']})")
    print(f"{'='*80}")
    print(f"\nLoss:")
    print(f"  Train Loss: {last_epoch['train_loss']:.4f}")
    print(f"  Val Loss:   {last_epoch['val_loss']:.4f}")
    
    print(f"\nTrain Metrics (Mean):")
    print(f"  Dice:      {last_epoch['train_metrics']['mean_dice']:.4f}")
    print(f"  IoU:       {last_epoch['train_metrics']['mean_iou']:.4f}")
    print(f"  Accuracy:  {last_epoch['train_metrics']['mean_accuracy']:.4f}")
    print(f"  Precision: {last_epoch['train_metrics']['mean_precision']:.4f}")
    print(f"  Recall:    {last_epoch['train_metrics']['mean_recall']:.4f}")
    print(f"  Pixel Acc: {last_epoch['train_metrics']['overall_pixel_acc']:.4f}")
    
    print(f"\nValidation Metrics (Mean):")
    print(f"  Dice:      {last_epoch['val_metrics']['mean_dice']:.4f}")
    print(f"  IoU:       {last_epoch['val_metrics']['mean_iou']:.4f}")
    print(f"  Accuracy:  {last_epoch['val_metrics']['mean_accuracy']:.4f}")
    print(f"  Precision: {last_epoch['val_metrics']['mean_precision']:.4f}")
    print(f"  Recall:    {last_epoch['val_metrics']['mean_recall']:.4f}")
    print(f"  Pixel Acc: {last_epoch['val_metrics']['overall_pixel_acc']:.4f}")
    
    # Find best metrics
    print(f"\n{'='*80}")
    print(f"Best Validation Metrics Across All Epochs:")
    print(f"{'='*80}")
    
    best_dice_epoch = max(epochs_data, key=lambda x: x['val_metrics']['mean_dice'])
    best_iou_epoch = max(epochs_data, key=lambda x: x['val_metrics']['mean_iou'])
    best_acc_epoch = max(epochs_data, key=lambda x: x['val_metrics']['mean_accuracy'])
    min_loss_epoch = min(epochs_data, key=lambda x: x['val_loss'])
    
    print(f"\nBest Dice:    {best_dice_epoch['val_metrics']['mean_dice']:.4f} (Epoch {best_dice_epoch['epoch']})")
    print(f"Best IoU:     {best_iou_epoch['val_metrics']['mean_iou']:.4f} (Epoch {best_iou_epoch['epoch']})")
    print(f"Best Accuracy:{best_acc_epoch['val_metrics']['mean_accuracy']:.4f} (Epoch {best_acc_epoch['epoch']})")
    print(f"Min Val Loss: {min_loss_epoch['val_loss']:.4f} (Epoch {min_loss_epoch['epoch']})")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--model', type=str, default='DeepLabV3+', help='Model name')
    parser.add_argument('--metrics_path', type=str, default='metrics/', help='Path to metrics directory')
    parser.add_argument('--summary_only', action='store_true', help='Only print summary without plotting')
    
    args = parser.parse_args()
    
    if args.summary_only:
        print_metrics_summary(args.model, args.metrics_path)
    else:
        plot_metrics(args.model, args.metrics_path)
