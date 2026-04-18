import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MULTI-CLASS LOSSES
# ============================================================================

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, n_classes=6):
        """
        Multi-Class Dice Loss for semantic segmentation.
        Args:
            smooth (float): Small value to avoid division by zero.
            n_classes (int): Number of classes.
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.n_classes = n_classes

    def forward(self, predictions, targets):
        """
        Forward pass for Multi-Class Dice Loss.
        Args:
            predictions (Tensor): Model predictions (logits), shape: (B, C, H, W)
            targets (Tensor): Ground truth labels (class indices), shape: (B, H, W)
        Returns:
            Loss (Tensor): Dice loss value.
        """
        # Convert logits to probabilities using softmax
        predictions = torch.softmax(predictions, dim=1)  # (B, C, H, W)
        
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), self.n_classes, -1)  # (B, C, H*W)
        targets = targets.view(targets.size(0), -1)  # (B, H*W)
        
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # (B, C, H*W)
        
        # Calculate Dice score per class
        intersection = (predictions * targets_one_hot).sum(dim=2)  # (B, C)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)  # (B, C)
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Return mean Dice loss across batch and classes
        return 1 - dice_score.mean()


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, n_classes=6):
        """
        Multi-Class Focal Loss for addressing class imbalance.
        Args:
            alpha (list or Tensor): Weighting factor for each class. If None, uses uniform weights.
            gamma (float): Focusing parameter.
            n_classes (int): Number of classes.
        """
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.n_classes = n_classes
        
        if alpha is None:
            self.alpha = torch.ones(n_classes)
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, predictions, targets):
        """
        Forward pass for Multi-Class Focal Loss.
        Args:
            predictions (Tensor): Model predictions (logits), shape: (B, C, H, W)
            targets (Tensor): Ground truth labels (class indices), shape: (B, H, W)
        Returns:
            Loss (Tensor): Focal loss value.
        """
        # Reshape for cross entropy loss
        B, C, H, W = predictions.shape
        predictions = predictions.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # (B*H*W, C)
        targets = targets.view(-1)  # (B*H*W,)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')  # (B*H*W,)
        
        # Calculate focal weight
        predictions_soft = torch.softmax(predictions, dim=1)  # (B*H*W, C)
        p_t = predictions_soft.gather(1, targets.view(-1, 1)).squeeze(1)  # (B*H*W,)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if available
        if self.alpha is not None:
            self.alpha = self.alpha.to(predictions.device)
            class_weight = self.alpha.gather(0, targets)  # (B*H*W,)
            focal_weight = focal_weight * class_weight
        
        # Return mean focal loss
        return (focal_weight * ce_loss).mean()


class MultiClassCombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, n_classes=6, smooth=1e-5):
        """
        Combined CrossEntropy + Dice Loss for multi-class segmentation.
        Args:
            alpha (float): Weight for CrossEntropy loss. (1-alpha) is the weight for Dice loss.
            n_classes (int): Number of classes.
            smooth (float): Small value to avoid division by zero in Dice loss.
        """
        super(MultiClassCombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(smooth=smooth, n_classes=n_classes)

    def forward(self, predictions, targets):
        """
        Forward pass for Combined Loss.
        Args:
            predictions (Tensor): Model predictions (logits), shape: (B, C, H, W)
            targets (Tensor): Ground truth labels (class indices), shape: (B, H, W)
        Returns:
            Loss (Tensor): Combined loss value.
        """
        ce_loss = self.ce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


# ============================================================================
# BINARY LOSSES (Legacy - for backward compatibility)
# ============================================================================

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Binary Dice Loss for semantic segmentation.
        Args:
            smooth (float): Small value to avoid division by zero.
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Forward pass for Binary Dice Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Dice loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            total_loss = 0
            for pred in predictions:
                total_loss += self._compute_dice_loss(pred, targets)
            return total_loss / len(predictions)
        else:
            # Single prediction
            return self._compute_dice_loss(predictions, targets)

    def _compute_dice_loss(self, predictions, targets):
        predictions = torch.sigmoid(predictions)  # Apply sigmoid for binary logits
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Binary Focal Loss for addressing class imbalance.
        Args:
            alpha (float): Weighting factor for positive class.
            gamma (float): Focusing parameter to reduce loss for well-classified examples.
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        Forward pass for Binary Focal Loss.
        Args:
            predictions (Tensor): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Focal loss value.
        """
        predictions = torch.sigmoid(predictions)  # Apply sigmoid for binary logits
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute the focal loss components
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class BinaryCombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5):
        """
        Combined BCE + Dice Loss for semantic segmentation.
        Args:
            alpha (float): Weight for BCE loss. (1-alpha) is the weight for Dice loss.
            smooth (float): Small value to avoid division by zero in Dice loss.
        """
        super(BinaryCombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        """
        Forward pass for Combined Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Combined loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            bce_loss = sum(self.bce(pred, targets) for pred in predictions) / len(predictions)
            dice_loss = self.dice(predictions, targets)
        else:
            # Single prediction
            bce_loss = self.bce(predictions, targets)
            dice_loss = self.dice(predictions, targets)

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, focal_alpha=0.25, smooth=1e-5):
        """
        Combined Focal + Dice Loss for semantic segmentation.
        Args:
            alpha (float): Weight for Focal loss. (1-alpha) is the weight for Dice loss.
            gamma (float): Focusing parameter for Focal loss.
            focal_alpha (float): Weighting factor for positive class in Focal loss.
            smooth (float): Small value to avoid division by zero in Dice loss.
        """
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=focal_alpha)
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        """
        Forward pass for Focal + Dice Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Combined loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            focal_loss = sum(self.focal(pred, targets) for pred in predictions) / len(predictions)
            dice_loss = self.dice(predictions, targets)
        else:
            # Single prediction
            focal_loss = self.focal(predictions, targets)
            dice_loss = self.dice(predictions, targets)

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss
