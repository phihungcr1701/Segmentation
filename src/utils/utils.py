import torch
import torchvision
import os
import json
from pathlib import Path

def save_checkpoint(state, filename='checkpoints/checkpoint.pth'):
    print("Saving checkpoint ...")
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def save_metrics(epoch, train_loss, val_loss, train_metrics_dict, val_metrics_dict, model_name, save_path='metrics/'):
    """
    Save training metrics to JSON file for visualization later.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss for the epoch
        val_loss: Validation loss for the epoch
        train_metrics_dict: Dictionary containing training metrics (output from check_accuracy)
        val_metrics_dict: Dictionary containing validation metrics (output from check_accuracy)
        model_name: Name of the model being trained
        save_path: Directory to save the metrics JSON file
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    metrics_file = os.path.join(save_path, f'{model_name}_metrics.json')
    
    # Create or load existing metrics
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {'epochs': []}
    
    # Convert numpy arrays to lists for JSON serialization
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_metrics': {
            'dice_scores': train_metrics_dict['dice_scores'].tolist(),
            'ious': train_metrics_dict['ious'].tolist(),
            'precisions': train_metrics_dict['precisions'].tolist(),
            'recalls': train_metrics_dict['recalls'].tolist(),
            'accuracies': train_metrics_dict['accuracies'].tolist(),
            'overall_pixel_acc': float(train_metrics_dict['overall_pixel_acc']),
            'mean_dice': float(train_metrics_dict['mean_dice']),
            'mean_iou': float(train_metrics_dict['mean_iou']),
            'mean_accuracy': float(train_metrics_dict['mean_accuracy']),
            'mean_precision': float(train_metrics_dict['mean_precision']),
            'mean_recall': float(train_metrics_dict['mean_recall'])
        },
        'val_metrics': {
            'dice_scores': val_metrics_dict['dice_scores'].tolist(),
            'ious': val_metrics_dict['ious'].tolist(),
            'precisions': val_metrics_dict['precisions'].tolist(),
            'recalls': val_metrics_dict['recalls'].tolist(),
            'accuracies': val_metrics_dict['accuracies'].tolist(),
            'overall_pixel_acc': float(val_metrics_dict['overall_pixel_acc']),
            'mean_dice': float(val_metrics_dict['mean_dice']),
            'mean_iou': float(val_metrics_dict['mean_iou']),
            'mean_accuracy': float(val_metrics_dict['mean_accuracy']),
            'mean_precision': float(val_metrics_dict['mean_precision']),
            'mean_recall': float(val_metrics_dict['mean_recall'])
        }
    }
    
    all_metrics['epochs'].append(epoch_metrics)
    
    # Save to JSON file
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_file}")
    
def load_checkpoint(filepath, model):
    print("Loading checkpoint ...")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
def check_accuracy(loader, model, device="cuda", verbose=True, logger=None):
    """
    Calculate accuracy metrics for multi-class segmentation:
    - Per-class Dice score
    - Per-class IoU (Intersection over Union)
    - Per-class Precision
    - Per-class Recall
    - Per-class Accuracy
    - Overall Pixel Accuracy
    
    Args:
        loader: DataLoader for evaluation
        model: Model to evaluate
        device: Device to use (cuda or cpu)
        verbose: Whether to print metrics
        logger: Logger object for logging metrics (optional)
    """
    model.eval()
    
    # Initialize metrics for each class
    dice_scores = torch.zeros(6).to(device)  # 6 classes
    ious = torch.zeros(6).to(device)
    precisions = torch.zeros(6).to(device)
    recalls = torch.zeros(6).to(device)
    accuracies = torch.zeros(6).to(device)
    
    total_correct = 0
    total_pixels = 0
    n_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # Shape: (B, H, W) with class indices 0-5
            
            # Get predictions: shape (B, 6, H, W)
            output = model(x)
            # Handle deep supervision
            if isinstance(output, list):
                output = output[-1]
            
            # Convert to class indices: argmax over 6 channels
            preds = torch.argmax(output, dim=1)  # Shape: (B, H, W) with values 0-5
            
            # Overall pixel accuracy
            correct = (preds == y).sum().item()
            total_correct += correct
            total_pixels += y.numel()
            
            # Calculate metrics for each class
            for class_idx in range(6):
                # Binary masks: True where prediction/target equals class_idx
                pred_class = (preds == class_idx).float()
                target_class = (y == class_idx).float()
                
                # Intersection and Union
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum() - intersection
                
                # True Positives, False Positives, False Negatives
                tp = intersection
                fp = pred_class.sum() - tp
                fn = target_class.sum() - tp
                
                # Dice Score: 2*TP/(2*TP + FP + FN)
                dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)
                dice_scores[class_idx] += dice
                
                # IoU: TP/(TP + FP + FN)
                iou = tp / (tp + fp + fn + 1e-8)
                ious[class_idx] += iou
                
                # Precision: TP/(TP + FP)
                precision = tp / (tp + fp + 1e-8)
                precisions[class_idx] += precision
                
                # Recall: TP/(TP + FN)
                recall = tp / (tp + fn + 1e-8)
                recalls[class_idx] += recall
                
                # Accuracy (per-class): TP/(TP + FP + FN)
                accuracy = tp / (tp + fp + fn + 1e-8)
                accuracies[class_idx] += accuracy
            
            n_samples += 1
    
    # Average metrics over all samples
    dice_scores /= n_samples
    ious /= n_samples
    precisions /= n_samples
    recalls /= n_samples
    accuracies /= n_samples
    
    # Calculate overall pixel accuracy
    overall_pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0
    
    # Print results only if verbose
    if verbose:
        output_fn = logger.info if logger else print
        
        output_fn("\nMetrics per class:")
        output_fn("Class\t\tDice\t\tIoU\t\tAccuracy\tPrecision\tRecall")
        output_fn("-" * 85)
        
        class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        
        for i in range(6):
            output_fn(f"{class_names[i]:<12} {dice_scores[i]:.4f}\t{ious[i]:.4f}\t{accuracies[i]:.4f}\t\t{precisions[i]:.4f}\t{recalls[i]:.4f}")
        
        # Print mean metrics
        output_fn("-" * 85)
        output_fn(f"Mean\t\t{dice_scores.mean():.4f}\t{ious.mean():.4f}\t{accuracies.mean():.4f}\t\t{precisions.mean():.4f}\t{recalls.mean():.4f}")
        output_fn(f"Overall Pixel Accuracy: {overall_pixel_acc:.4f}")
    
    return {
        'dice_scores': dice_scores.cpu().numpy(),
        'ious': ious.cpu().numpy(),
        'precisions': precisions.cpu().numpy(),
        'recalls': recalls.cpu().numpy(),
        'accuracies': accuracies.cpu().numpy(),
        'overall_pixel_acc': overall_pixel_acc,
        'mean_dice': dice_scores.mean().item(),
        'mean_iou': ious.mean().item(),
        'mean_accuracy': accuracies.mean().item(),
        'mean_precision': precisions.mean().item(),
        'mean_recall': recalls.mean().item()
    }
        
def save_predictions_as_imgs(loader, model, epoch, mod=50, folder='results/', device='cuda'):
    print('Saving predictions as images ...')
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx % mod == 0:
            x = x.to(device=device)
            y = y.to(device=device)
            with torch.no_grad():
                output = model(x)
                # Handle both deep supervision and single output cases
                if isinstance(output, list):
                    # Take the final output (most refined prediction)
                    output = output[-1]
                
                # Convert multi-class output to class indices
                # output shape: (B, 6, H, W) -> preds shape: (B, H, W)
                preds = torch.argmax(output, dim=1)  # Get class with max probability
                # Normalize class indices 0-5 to 0-255 for visualization
                preds = (preds.float() / 5.0 * 255).unsqueeze(1).byte()  # (B, 1, H, W)
            
            def make_folder(path):
                # Check if the folder exists, and if not, create it
                if not os.path.exists(path):
                    os.makedirs(path)
            
            # Convert ground truth mask to saveable format
            # y shape: (B, H, W) with class indices 0-5
            y_normalized = (y.float() / 5.0 * 255).unsqueeze(1).byte()  # (B, 1, H, W)
            
            # Save images
            if epoch == 0:
                make_folder(f"{folder}pred/epoch_{epoch}/")
                torchvision.utils.save_image(preds, f"{folder}pred/epoch_{epoch}/pred_{idx}.jpg")
                make_folder(f"{folder}true/epoch_{epoch}/")
                torchvision.utils.save_image(y_normalized, f"{folder}true/epoch_{epoch}/true_{idx}.jpg")
            make_folder(f"{folder}image/epoch_{epoch}/")
            torchvision.utils.save_image(x, f"{folder}image/epoch_{epoch}/image{idx}.jpg")
            
def test():
    from src.datasets.dataset import PanNukeDataset
    from torch.utils.data import DataLoader
    
    from src.models.UNet import UNet
    
    DEVICE = 'cuda'
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UNet(in_channels=3, out_channels=6).to(DEVICE)
    for epoch in range(1):
        for i, (image, mask) in enumerate(dataloader):
            #check_accuracy(dataloader, model, device=DEVICE)
            
            save_predictions_as_imgs(dataloader, model, epoch=epoch, folder='results/', device=DEVICE)
            
            break
    
if __name__ == '__main__':
    test()
