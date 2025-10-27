import os
import argparse
import json
import warnings
import shutil
import torch
import torch.nn as nn
import logging

from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
from torch.utils.data import DataLoader
from foundation.model import DINOClassifier, create_multiple_classifiers, AllClassifiers, FocalLoss, LabelSmoothingCrossEntropy
from utils.utils import DEFAULT_IMAGE_SIZE, SEED, FUS_STRUCTS, FPDB_BRAIN_PLANES, SaveBestModel, save_model, save_plots
# Removed import of seg train/validate, will define cls versions below

from datahandler import get_fetalus_dataloaders
from engine import train, validate, get_lr_scheduler, EarlyStopping


def plot_confusion_matrix(cm, classes, out_dir, title='Confusion Matrix'):
    """
    Plot and save confusion matrix as image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16})  # Large font for cells
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_grid(dataset, dataloader, backbone_model, classifier, classes, output_dir, device='cpu', max_samples=16, backbone_type='vit'):
    """
    Create a grid visualization of predictions showing input images with ground truth and predictions.

    Args:
        dataset: The dataset object
        dataloader: DataLoader for the dataset
        backbone_model: The backbone model (DINOv2/v3)
        classifier: The classifier model (AllClassifiers with 'temp' key)
        classes: List of class names
        output_dir: Directory to save the visualization
        device: Device to run inference on
        max_samples: Maximum number of samples to visualize (default: 16)
    """
    backbone_model.eval()
    classifier.eval()

    all_images = []
    all_labels = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Get features and predictions
            features = backbone_model.backbone_model(batch_images)
            if backbone_type == 'convnext':
                if len(features.shape) == 4:
                    features = [torch.mean(features, dim=[2, 3])]
                else:
                    features = [features]
            outputs = classifier(features)['temp']  # Get the single classifier output

            preds = torch.argmax(outputs, dim=1)
            confidences = torch.nn.functional.softmax(outputs, dim=1)

            all_images.extend(batch_images.cpu())
            all_labels.extend(batch_labels.cpu())
            all_preds.extend(preds.cpu())
            all_confidences.extend(confidences.cpu())

            if len(all_images) >= max_samples:
                break

    # Keep only the first max_samples
    all_images = all_images[:max_samples]
    all_labels = all_labels[:max_samples]
    all_preds = all_preds[:max_samples]
    all_confidences = all_confidences[:max_samples]

    # Create visualization grid
    n_cols = 4
    n_rows = (len(all_images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(len(all_images)):
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]

        # Denormalize image (reverse ImageNet normalization if needed)
        img = all_images[idx]
        if img.shape[0] == 3:  # RGB image, denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
        img = torch.clamp(img, 0, 1)  # Clamp to valid range

        # Convert to numpy and transpose to HWC
        img_np = img.permute(1, 2, 0).numpy()

        # For RGB images, convert to grayscale to avoid color artifacts
        if img.shape[0] == 3:
            img_np = np.mean(img_np, axis=2)

        ax.imshow(img_np, cmap='gray')
        ax.axis('off')

        # Get prediction info
        true_label = all_labels[idx].item()
        pred_label = all_preds[idx].item()
        confidence = all_confidences[idx][pred_label].item()

        # Color based on correctness
        color = 'green' if true_label == pred_label else 'red'

        title = f'GT: {classes[true_label]}\nPred: {classes[pred_label]}\nConf: {confidence:.2f}'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold' if true_label != pred_label else 'normal')

    # Hide empty subplots
    for idx in range(len(all_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary
    correct_predictions = sum(1 for gt, pred in zip(all_labels, all_preds) if gt == pred)
    accuracy = correct_predictions / len(all_labels)
    print(f"Sample prediction grid - Accuracy on shown samples: {accuracy:.2f} ({correct_predictions}/{len(all_labels)})")


# ==========================================================================================
# ==================================== GLOBAL SETTINGS =====================================
# Suppress warnings
# warnings.filterwarnings("ignore", message="xFormers is not available")

# Set random seed for reproducibility
torch.manual_seed(SEED)

# Set random seed for CUDA and MPS if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
# ==========================================================================================


# ========================================================================================== 
# =================================== CLI ARGS PARSER ======================================
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="root of the project")
parser.add_argument("--datasets", type=str, required=True, help="path to the datasets")
parser.add_argument("--exp-id", type=int, help="index of experiment to run from config")
parser.add_argument('--debug', action='store_true', help='debug mode, reduces dataset size for faster testing')
args = parser.parse_args()
# ==========================================================================================


# ==========================================================================================
# =================================== LOGGING SYSTEM =======================================
# Console and logging setup
console = Console()
log = logging.getLogger("rich")
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
    force=True  # Force reconfiguration of logging
)
# Ensure logs are flushed immediately
import sys
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
log.addHandler(log_handler)
# ==========================================================================================


if __name__ == '__main__':
    log.info("Script started")

    # ==========================================================================================
    # ===================================== INITIALIZATION =====================================
    # Print command line arguments
    console.print(Panel(
        "\n".join([f"[bold magenta]{k}[/bold magenta]: [white]{v}[/white]" for k, v in vars(args).items()]),
        title="[magenta]Command Line Arguments[/magenta]",
        style="bold magenta"
    ))

    # Load experiment config
    config_path = os.path.join(args.root, 'configs/experiments.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config file not found at {config_path}")
    with open(config_path) as f:
        experiments = json.load(f)
    
    # Handle exp-id argument robustly
    if args.exp_id is not None:
        if not (0 <= args.exp_id < len(experiments)):
            raise ValueError(f"exp-id {args.exp_id} is out of range (0-{len(experiments)-1})")
        exp_id = args.exp_id
    else:
        exp_id = 0
    config = experiments[exp_id]

    # fallback for config parameters
    config.setdefault('image_size', DEFAULT_IMAGE_SIZE)
    config.setdefault('batch_size', 16)  # default batch size

    # classes
    classes = FPDB_BRAIN_PLANES

    # Get backbone type for feature processing
    backbone_type = config['dino'].get('backbone_type', 'vit')

    # Load grid search configurations early (before using in panels)
    grid_config = config.get('grid_search', {})
    learning_rates_list = grid_config.get('learning_rates', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    n_blocks_list = grid_config.get('n_blocks', [1, 2, 4])
    loss_types_list = grid_config.get('loss_types', ['ce', 'focal', 'label_smooth'])
    schedulers_list = grid_config.get('schedulers', ['step', 'cosine', 'plateau'])

    # For now, use global batch_size and scheduler (configurable for future experiments)
    scheduler_type = schedulers_list[0] if schedulers_list else 'cosine'  # Use first scheduler as default

    # Create output directory of the training experiment
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    exp_name = config['datasets'][0] if len(config['datasets']) == 1 else ''.join([d[0].lower() for d in config['datasets']])
    out_dir = os.path.join(args.root, 'outputs', f"{config['dino']['version']}_{exp_name}_{timestamp}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Save config copy
    config_copy = config.copy()
    config_copy['task'] = 'cls'
    config_copy['classes'] = classes
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config_copy, f, indent=2)

    # Print experiment config parameters in a formatted way
    console.print(Panel(
        "\n".join([
            f"[bold yellow]Experiment directory[/bold yellow]: [white]{out_dir}[/white]",
            f"[bold yellow]Seed[/bold yellow]: [white]{SEED}[/white]\n",
            f"[bold yellow]DINO version[/bold yellow]: [white]{config['dino']['version']}[/white]",
            f"[bold yellow]DINO backbone type[/bold yellow]: [white]{config['dino']['backbone_type']}[/white]",
            f"[bold yellow]DINO backbone size[/bold yellow]: [white]{config['dino']['backbone_size']}[/white]",
            f"[bold yellow]DINO feature layers[/bold yellow]: [white]{config['dino']['intermediate_layers']}[/white]",
            f"[bold yellow]DINO backbone fine-tuning [/bold yellow]: [white]{config['dino']['fine_tune']}[/white]",
            f"[bold yellow]Datasets[/bold yellow]: [white]{', '.join(config['datasets'])}[/white]",
            f"[bold yellow]Image Size[/bold yellow]: [white]{config['image_size']}[/white]",
            f"[bold yellow]Epochs[/bold yellow]: [white]{config['epochs']}[/white]",
            f"[bold yellow]Batch size[/bold yellow]: [white]{config['batch_size']}[/white]",
            f"[bold yellow]Weight decay[/bold yellow]: [white]{config['weight_decay']}[/white]",
            f"[bold yellow]Early stopping patience[/bold yellow]: [white]{config['early_stop_patience']}[/white]",
            f"[bold yellow]\n📊 GRID SEARCH CONFIGURATIONS 📊[/bold yellow]",
            f"[bold yellow]N. blocks[/bold yellow]: [white]{n_blocks_list}[/white]",
            f"[bold yellow]Learning rates[/bold yellow]: [white]{learning_rates_list}[/white]",
            f"[bold yellow]Loss functions[/bold yellow]: [white]{loss_types_list}[/white]",
            f"[bold yellow]Schedulers[/bold yellow]: [white]{schedulers_list}[/white]",
            f"[bold yellow]Scheduler (current)[/bold yellow]: [white]{scheduler_type.upper()}[/white]",
            f"[bold yellow]Total combinations[/bold yellow]: [white]{len(learning_rates_list) * len(n_blocks_list) * 2 * len(loss_types_list)} (CLS + AvgPool)[/white]",
        ]),
        title=f"[yellow]Experiment [bold]{exp_name}[/bold] (ID: {exp_id})[/yellow]",
        style="bold yellow"
    ))
    # ==========================================================================================
    

    # ==========================================================================================
    # ======================================== DATASET =========================================
    log.info("Loading dataset...")

    # get datasets classes
    eval_augmentation = config.get('eval_augmentation', False)
    fus_train, fus_val, fus_test = get_fetalus_dataloaders(
        root=args.root,
        data_path=args.datasets,
        datasets=config['datasets'],
        image_size=config['image_size'],
        supervised=True,
        debug=args.debug,
        task='cls',
        eval_augmentation=eval_augmentation)
    
    # Create DataLoaders
    train_loader = DataLoader(fus_train, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(fus_val, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(fus_test, batch_size=config['batch_size'], shuffle=False)

    num_classes = len(classes)
    
    # Print dataset info using rich Panel
    console.print(Panel(
        "\n".join([
            f"[bold cyan]Image size[/bold cyan]: [white]{config['image_size']}[/white]",
            f"[bold cyan]Classes[/bold cyan]: [white]{classes} ({num_classes})[/white]",
            f"[bold cyan]Dataset size[/bold cyan]: [white]{len(fus_train) + len(fus_val) + len(fus_test)}[/white]",
            f"[bold cyan]Train set size[/bold cyan]: [white]{len(fus_train)}[/white]",
            f"[bold cyan]Validation set size[/bold cyan]: [white]{len(fus_val)}[/white]",
            f"[bold cyan]Test set size[/bold cyan]: [white]{len(fus_test)}[/white]",
        ]),
        title="[cyan]Dataset[/cyan]",
        style="bold cyan"
    ))
    # ==========================================================================================


    # ==========================================================================================
    # ========================================= MODEL ==========================================
    # Set device based on availability
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    log.info(f"Using device: {device}")
    
    # Grid search configurations loaded from config
    grid_config = config.get('grid_search', {})
    learning_rates_list = grid_config.get('learning_rates', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    n_blocks_list = grid_config.get('n_blocks', [1, 2, 4])
    loss_types_list = grid_config.get('loss_types', ['ce', 'focal', 'label_smooth'])
    schedulers_list = grid_config.get('schedulers', ['step', 'cosine', 'plateau'])

    # For now, use global batch_size and scheduler (configurable for future experiments)
    scheduler_type = schedulers_list[0] if schedulers_list else 'cosine'  # Use first scheduler as default
    
    # Initialize the single backbone model (backbone shared across all classifiers)
    backbone_model = DINOClassifier(root=args.root,
                                    nc=num_classes,
                                    image_size=config['image_size'],
                                    config=config['dino'],
                                    device=device)
    backbone_model.to(device)

    if args.debug:
        log.debug("Backbone model summary")
        summary(
            backbone_model,
            (1, 3, config['image_size'][0], config['image_size'][1]),
            col_names=('input_size', 'output_size', 'num_params'),
            row_settings=['var_names']
        )

    # Create dummy input to get backbone features for multiple classifiers
    dummy_input = torch.randn(1, 3, config['image_size'][0], config['image_size'][1]).to(device)
    sample_output = backbone_model.backbone_model(dummy_input)

    # Create multiple classifiers with comprehensive grid search (architectures + loss functions)
    linear_classifiers, optim_param_groups, loss_functions = create_multiple_classifiers(
        sample_output=sample_output,
        n_last_blocks_list=n_blocks_list,
        learning_rates=learning_rates_list,
        batch_size=config['batch_size'],
        num_classes=num_classes,
        loss_types=loss_types_list,
        backbone_type=config['dino'].get('backbone_type', 'vit')
    )

    # Move classifiers to device if using CUDA
    if torch.cuda.is_available():
        linear_classifiers = linear_classifiers.cuda()

    log.info(f"Created {len(linear_classifiers.classifiers_dict)} classifier configurations")
    for name in linear_classifiers.classifiers_dict.keys():
        log.debug(f"Classifier: {name}")

    # Optimizer with different LRs for each classifier (as per DINOv3)
    optimizer = torch.optim.AdamW(optim_param_groups, weight_decay=config['weight_decay'])
    log.debug(f"Optimizer: AdamW with weight decay: {config['weight_decay']} and multiple learning rates")

    # For classification, always use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    log.debug("Using Cross Entropy Loss")

    # exit(0) # temporary exit to avoid running training during testing of the script

    # ==========================================================================================
    # ================================== LR SCHEDULER & EARLY STOPPING ==========================
    # Calculate epoch length for advanced schedulers
    epoch_length = len(train_loader)  # Iterations per epoch
    max_iter = config['epochs'] * epoch_length

    # Use CosineAnnealing scheduler as per DINOv3 best practices
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-6)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stop_patience', 10),
        min_delta=config.get('early_stop_min_delta', 0.001),
        verbose=True
    )

    # Track best classifier across all configurations
    best_classifier_name = None
    best_accuracy = -1.0

    # ==========================================================================================
    # ===================================== TRAINING LOOP ======================================
    log.info("Starting training loop with multiple classifiers...")
    log.info(f"Training on {len(train_loader)} batches per epoch with {len(linear_classifiers.classifiers_dict)} classifier configurations")
    log.info(f"Batch size: {config['batch_size']}, Learning rate scaled automatically per classifier")
    log.info(f"Using CosineAnnealingLR scheduler (T_max: {max_iter}, eta_min: 1e-6)")

    # Initialize classes for saving best models (will be used for the best classifier)
    save_best_model = SaveBestModel()

    # Initialize lists to store training and validation loss and metrics per epoch
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    # Track metrics per classifier for grid search analysis
    classifier_metrics = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                         for name in linear_classifiers.classifiers_dict.keys()}

    iteration = 0
    for epoch in range(config['epochs']):
        log.info(f"🟢 EPOCH {epoch + 1}/{config['epochs']} - Starting training phase...")

        # ======================== TRAINING PHASE ========================
        backbone_model.train()
        linear_classifiers.train()

        total_train_loss = 0.0
        total_train_correct = 0.0
        total_train_samples = 0

        log.info(f"   Training on {len(train_loader)} batches...")
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:  # Log ogni 10 batch
                log.info(f"   Batch {batch_idx+1}/{len(train_loader)} - Processing...")

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Extract features with frozen backbone
            with torch.no_grad():
                features = backbone_model.backbone_model(batch_data)

            if backbone_type == 'convnext':
                if len(features.shape) == 4:
                    features = [torch.mean(features, dim=[2, 3])]
                else:
                    features = [features]

            # Forward pass through all classifiers and compute loss using their specific loss functions
            outputs = linear_classifiers(features)
            losses = {f"loss_{k}": loss_functions[k](v, batch_labels) for k, v in outputs.items()}
            loss = sum(losses.values()) / len(outputs)  # Average loss across classifiers for proper scaling

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training metrics for all classifiers combined
            batch_size = batch_labels.size(0)
            batch_acc = 0.0
            for k in outputs.keys():
                acc_k = (torch.argmax(outputs[k], dim=1).eq(batch_labels)).float().mean().item()
                batch_acc += acc_k
            batch_acc /= len(outputs)
            total_train_correct += batch_acc * batch_size
            total_train_loss += loss.item()
            total_train_samples += batch_size

            iteration += 1
            scheduler.step()

            # Accumulate training metrics per classifier for grid search analysis
            # Note: Since all classifiers are trained simultaneously, we store the combined average
            # as a proxy for individual classifier training progress (updated at epoch end)

            # Log intermediate progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log.info(f"   📊 Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

            # Log intermediate progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log.info(f"   📊 Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        log.info("   ✅ Training phase completed")

        # Calculate average training metrics across all classifiers
        avg_train_loss = total_train_loss / len(train_loader)
        # Fixed: calculate accuracy as total correct over total samples
        avg_train_acc = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0

        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)

        # Store average training metrics for ALL classifiers at epoch end (they train simultaneously)
        for classifier_name in classifier_metrics.keys():
            classifier_metrics[classifier_name]['train_loss'].append(avg_train_loss)
            classifier_metrics[classifier_name]['train_acc'].append(avg_train_acc)

        log.info(f"🔵 EPOCH {epoch + 1} - Starting validation phase on {len(val_loader)} batches...")

        # ======================== VALIDATION PHASE ========================
        backbone_model.eval()
        linear_classifiers.eval()

        epoch_val_results = {}  # Store results for each classifier

        with torch.no_grad():
            for batch_idx, (batch_data, batch_labels) in enumerate(val_loader):
                if batch_idx % 10 == 0:  # Log ogni 10 batch di validation
                    log.info(f"   Validation batch {batch_idx+1}/{len(val_loader)} - Evaluating...")

                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                features = backbone_model.backbone_model(batch_data)
                if backbone_type == 'convnext':
                    if len(features.shape) == 4:
                        features = [torch.mean(features, dim=[2, 3])]
                    else:
                        features = [features]
                outputs = linear_classifiers(features)

                for classifier_name, classifier_output in outputs.items():
                    if classifier_name not in epoch_val_results:
                        epoch_val_results[classifier_name] = {
                            'loss': 0.0, 'correct': 0, 'total': 0, 'preds': [], 'labels': []
                        }

                    loss = loss_functions[classifier_name](classifier_output, batch_labels)
                    preds = torch.argmax(classifier_output, dim=1)

                    epoch_val_results[classifier_name]['loss'] += loss.item()
                    epoch_val_results[classifier_name]['correct'] += preds.eq(batch_labels).sum().item()
                    epoch_val_results[classifier_name]['total'] += batch_labels.size(0)
                    epoch_val_results[classifier_name]['preds'].extend(preds.cpu().numpy())
                    epoch_val_results[classifier_name]['labels'].extend(batch_labels.cpu().numpy())

        log.info("   ✅ Validation phase completed")

        # Process validation results for each classifier
        best_epoch_accuracy = -1.0
        best_epoch_classifier = None
        log.info("   🔍 Analyzing validation results across all classifiers...")

        for classifier_name, results in epoch_val_results.items():
            val_loss = results['loss'] / len(val_loader)
            val_accuracy = results['correct'] / results['total']

            classifier_metrics[classifier_name]['val_loss'].append(val_loss)
            classifier_metrics[classifier_name]['val_acc'].append(val_accuracy)

            if val_accuracy > best_epoch_accuracy:
                best_epoch_accuracy = val_accuracy
                best_epoch_classifier = classifier_name

        log.info(f"   📈 Epoch best: {best_epoch_accuracy:.4f} ({best_epoch_classifier})")

        # Update global best classifier
        updated_global = False
        if best_epoch_accuracy > best_accuracy:
            best_accuracy = best_epoch_accuracy
            best_classifier_name = best_epoch_classifier
            updated_global = True
            log.info(f"   🎯 NEW GLOBAL BEST: {best_accuracy:.4f} ({best_classifier_name}) - Saving checkpoint...")

            # Save the best classifier state
            best_classifier_state = {
                'backbone_state_dict': backbone_model.state_dict(),
                'classifier_state_dict': linear_classifiers.classifiers_dict[best_classifier_name].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'best_classifier_name': best_classifier_name,
                'epoch': epoch
            }
            torch.save(best_classifier_state, os.path.join(out_dir, 'best_classifier.pth'))
            log.info("   💾 Checkpoint saved successfully")
        else:
            log.info(f"   🔄 No improvement - Global best remains: {best_accuracy:.4f} ({best_classifier_name})")

        # Use weighted average validation metrics for early stopping
        avg_val_loss = sum(results['loss'] / len(val_loader) for results in epoch_val_results.values()) / len(epoch_val_results)
        avg_val_acc = sum(results['correct'] / results['total'] for results in epoch_val_results.values()) / len(epoch_val_results)

        valid_loss.append(avg_val_loss)
        valid_acc.append(avg_val_acc)

        # Check early stopping based on best accuracy
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            log.warning(f"Early stopping triggered at epoch {epoch + 1} - No improvement in {early_stopping.patience} epochs")
            break

        # Log comprehensive metrics
        log.info("🎯 EPOCH SUMMARY:")
        print(f"   Training Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.4f}")
        print(f"   Best Val Acc: {best_epoch_accuracy:.4f} ({best_epoch_classifier})")
        print(f"   Avg Val Loss: {avg_val_loss:.4f}, Avg Val Acc: {avg_val_acc:.4f}")
        print(f"   Global Best: {best_accuracy:.4f} ({best_classifier_name})")
        print('-' * 90)

    # Final comprehensive training summary
    log.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    log.info("=" * 100)
    log.info(f"📈 Final Results Summary:")
    log.info(f"   └─ Best Classifier: {best_classifier_name}")
    log.info(f"   └─ Best Validation Accuracy: {best_accuracy:.4f}")
    log.info(f"   └─ Total Configurations Tested: {len(linear_classifiers.classifiers_dict)}")
    log.info(f"   └─ Training Epochs Completed: {len(train_loss)}")
    log.info(f"   └─ Final Training Loss: {train_loss[-1]:.4f}")
    log.info(f"   └─ Final Training Accuracy: {train_acc[-1]:.4f}")
    log.info(f"   └─ Final Validation Loss: {valid_loss[-1]:.4f}")
    log.info(f"   └─ Final Validation Accuracy: {valid_acc[-1]:.4f}")
    log.info("=" * 100)

    # Save classifier metrics for analysis
    with open(os.path.join(out_dir, 'classifier_grid_search_metrics.json'), 'w') as f:
        json.dump(classifier_metrics, f, indent=2)
    # ==================================================================================


    # =================================== SAVE BEST MODEL RESULTS ==================================
    log.info("Saving best model results and comprehensive grid search analysis...")

    # Create grid_search subdirectory for detailed results of all configurations
    grid_search_dir = os.path.join(out_dir, 'grid_search')
    os.makedirs(grid_search_dir, exist_ok=True)

    # Save comprehensive grid search results
    log.info(f"Saving grid search analysis to {grid_search_dir}")

    # Save classifier metrics JSON
    with open(os.path.join(grid_search_dir, 'all_classifier_metrics.json'), 'w') as f:
        json.dump(classifier_metrics, f, indent=2)

    # Save final best classifier summary
    grid_search_summary = {
        'best_classifier': best_classifier_name,
        'best_accuracy': best_accuracy,
        'total_configurations_tested': len(linear_classifiers.classifiers_dict),
        'grid_search_parameters': {
            'learning_rates': learning_rates_list,
            'n_blocks': n_blocks_list,
            'loss_types': loss_types_list,
            'schedulers': schedulers_list,
            'current_scheduler': scheduler_type,
            'current_batch_size': config['batch_size']
        },
        'training_epochs_completed': len(train_loss),
        'best_classifier_found_at_epoch': epoch if best_classifier_name else None,
        'early_stopping_triggered': early_stopping.early_stop
    }

    with open(os.path.join(grid_search_dir, 'grid_search_summary.json'), 'w') as f:
        json.dump(grid_search_summary, f, indent=2)

    # Save detailed results for each classifier configuration
    best_classifier_dir = os.path.join(grid_search_dir, 'best_classifier')
    os.makedirs(best_classifier_dir, exist_ok=True)

    # Save the best classifier model and its training curves (from final epoch)
    if best_classifier_name:
        log.info(f"Saving best classifier results: {best_classifier_name}")

        # Get the best classifier's final performance metrics
        best_metrics = classifier_metrics[best_classifier_name]
        final_train_loss = best_metrics['train_loss'][-1] if best_metrics['train_loss'] else 0
        final_train_acc = best_metrics['train_acc'][-1] if best_metrics['train_acc'] else 0
        final_val_loss = best_metrics['val_loss'][-1] if best_metrics['val_loss'] else 0
        final_val_acc = best_metrics['val_acc'][-1] if best_metrics['val_acc'] else 0

        # Since we train all classifiers simultaneously, create synthetic curves showing
        # the evolution of the best classifier across epochs
        epochs_range = list(range(1, len(best_metrics['val_acc']) + 1))
        best_train_acc_curve = [best_metrics['train_acc'][i] if i < len(best_metrics['train_acc']) else best_metrics['train_acc'][-1] for i in range(len(epochs_range))]
        best_val_acc_curve = best_metrics['val_acc']
        best_train_loss_curve = [best_metrics['train_loss'][i] if i < len(best_metrics['train_loss']) else best_metrics['train_loss'][-1] for i in range(len(epochs_range))]
        best_val_loss_curve = best_metrics['val_loss']

        # Save plots showing the training curves of the best classifier across epochs
        save_plots(best_train_acc_curve, best_val_acc_curve, best_train_loss_curve, best_val_loss_curve,
                  best_train_acc_curve, best_val_acc_curve, best_classifier_dir, task='cls')

        # Removed individual classifier performance files to avoid sub-folders for each configuration

        # Save final best model
        final_model_state = {
            'backbone_state_dict': backbone_model.state_dict(),
            'best_classifier_name': best_classifier_name,
            'best_classifier_state_dict': linear_classifiers.classifiers_dict[best_classifier_name].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'config': config_copy,
            'epoch': len(train_loss)
        }
        torch.save(final_model_state, os.path.join(best_classifier_dir, 'final_model.pth'))
        log.info(f"✅ Saved best classifier model and training curves to {best_classifier_dir}")
    else:
        log.warning("No best classifier found - no model saved")

    # ==================================================================================


    # ==================================================================================
    # =================================== TEST PHASE ===================================
    log.info("Starting test phase...")

    # Skip test phase if test set is empty
    if len(fus_test) == 0:
        log.info("Test set is empty, skipping test phase")
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': valid_loss,
            'val_acc': valid_acc,
            'test_loss': None,
            'test_acc': None,
            'test_f1_score': None,
            'per_class_accuracy': None,
            'per_class_f1': None,
            'best_classifier': best_classifier_name,
            'best_accuracy': best_accuracy
        }
    else:
        # Load the best classifier found during training
        if best_classifier_name:
            log.info(f"Loading best classifier: {best_classifier_name} (val acc: {best_accuracy:.4f})")
            checkpoint = torch.load(os.path.join(out_dir, 'best_classifier.pth'), weights_only=False)

            # Load backbone and best classifier
            backbone_model.load_state_dict(checkpoint['backbone_state_dict'])
            best_classifier = AllClassifiers({'temp': linear_classifiers.classifiers_dict[best_classifier_name]})
            best_classifier.classifiers_dict['temp'].load_state_dict(checkpoint['classifier_state_dict'])

            backbone_model.eval()
            best_classifier.eval()

            # Compute predictions and labels for confusion matrix
            all_preds = []
            all_labels = []
            test_loss_accum = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data in test_loader:
                    pixel_values, labels = data[0].to(device), data[1].to(device)
                    features = backbone_model.backbone_model(pixel_values)
                    if backbone_type == 'convnext':
                        if len(features.shape) == 4:
                            features = [torch.mean(features, dim=[2, 3])]
                        else:
                            features = [features]
                    outputs = best_classifier(features)['temp']  # Get the single classifier output
                    
                    # Compute loss using the specific loss function of the best classifier
                    loss = loss_functions[best_classifier_name](outputs, labels)
                    test_loss_accum += loss.item()

                    # Get predictions
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # Accumulate accuracy
                    test_correct += preds.eq(labels).sum().item()
                    test_total += labels.size(0)
            
            test_loss = test_loss_accum / len(test_loader)
            test_acc = test_correct / test_total

            # Compute F1-score early (before saving report)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            # Create and save sample prediction grid for manual inspection
            log.info("Creating sample prediction grid for visual inspection...")
            try:
                create_prediction_grid(fus_test, test_loader, backbone_model, best_classifier, classes, best_classifier_dir, device=device, max_samples=20, backbone_type=backbone_type)
                log.info("Sample prediction grid saved successfully")
            except Exception as e:
                log.warning(f"Could not create prediction grid: {e}")

            # Print confusion matrix and classification report
            cm = confusion_matrix(all_labels, all_preds)
            print("\nConfusion Matrix:")
            print(cm)

            # Save confusion matrix image in best classifier directory
            plot_confusion_matrix(cm, classes, best_classifier_dir, title=f'Brain Planes Classification ({best_classifier_name})')

            # Generate classification report
            class_report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
            print("\nClassification Report:")
            print(class_report)

            # Compute per-class metrics
            class_report_dict = classification_report(all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True)
            per_class_accuracy = {classes[i]: class_report_dict[classes[i]]['recall'] for i in range(len(classes))}
            per_class_f1 = {classes[i]: class_report_dict[classes[i]]['f1-score'] for i in range(len(classes))}

            # Save classification report to best classifier directory
            with open(os.path.join(best_classifier_dir, 'classification_report.txt'), 'w') as f:
                f.write(f"Brain Planes Classification Report - {best_classifier_name}\n")
                f.write("="*70 + "\n\n")
                f.write(f"Best Classifier: {best_classifier_name}\n")
                f.write(f"Validation Accuracy: {best_accuracy:.4f}\n")
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Test F1 Score: {f1:.4f}\n\n")
                f.write(f"Per-Class Accuracy: {per_class_accuracy}\n")
                f.write(f"Per-Class F1: {per_class_f1}\n\n")
                f.write(class_report)

            # Save final metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': valid_loss,
                'val_acc': valid_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1_score': float(f1),
                'per_class_accuracy': per_class_accuracy,
                'per_class_f1': per_class_f1,
                'best_classifier': best_classifier_name,
                'best_val_accuracy': best_accuracy
            }

            log.info(f"Test Results - Classifier: {best_classifier_name}")
            log.info(f"Test Accuracy: {test_acc:.4f}, Test F1: {f1:.4f}")
        else:
            log.warning("No best classifier found, skipping test phase")
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': valid_loss,
                'val_acc': valid_acc,
                'test_loss': None,
                'test_acc': None,
                'test_f1_score': None,
                'per_class_accuracy': None,
                'per_class_f1': None,
                'best_classifier': None,
                'best_accuracy': None
            }

    log.info("Saving metrics to JSON file...")
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    log.info("Test phase completed.")
    # ==================================================================================
    
    
    log.info("Task completed successfully.")
