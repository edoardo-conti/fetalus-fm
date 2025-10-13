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

from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
from torch.utils.data import DataLoader
from foundation.model import DINOClassifier
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
parser.add_argument('--fine-tune', dest='fine_tune', action='store_true', help='whether to fine tune the backbone or not')
parser.add_argument('--debug', action='store_true', help='debug mode, reduces dataset size for faster testing')
args = parser.parse_args()
# ==========================================================================================


# ========================================================================================== 
# =================================== LOGGING SYSTEM =======================================
# Console and logging setup
console = Console()
log = logging.getLogger("rich")
# logging.basicConfig(
#     level=logging.DEBUG if args.debug else logging.INFO,
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler()]
# )
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

    # classes
    classes = FPDB_BRAIN_PLANES

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
            f"[bold yellow]DINO backbone[/bold yellow]: [white]{config['dino']['backbone_size']}[/white]",
            f"[bold yellow]DINO intermediate layers[/bold yellow]: [white]{config['dino']['intermediate_layers']}[/white]",
            f"[bold yellow]Datasets[/bold yellow]: [white]{', '.join(config['datasets'])}[/white]",
            f"[bold yellow]Image Size[/bold yellow]: [white]{config['image_size']}[/white]",
            f"[bold yellow]Epochs[/bold yellow]: [white]{config['epochs']}[/white]",
            f"[bold yellow]Loss Function[/bold yellow]: [white]{config['loss']}[/white]",
            f"[bold yellow]Batch size[/bold yellow]: [white]{config['batch_size']}[/white]",
            f"[bold yellow]Learning rate[/bold yellow]: [white]{config['lr']}[/white]",
            f"[bold yellow]Weight decay[/bold yellow]: [white]{config['weight_decay']}[/white]",
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
    
    # Initialize the model
    model = DINOClassifier(nc=num_classes,
                               image_size=config['image_size'],
                               config=config['dino'],
                               fine_tune=args.fine_tune,
                               device=device)
    model.to(device)

    if args.debug:
        log.debug("Model summary")
        summary(
            model, 
            (1, 3, config['image_size'][0], config['image_size'][1]),
            col_names=('input_size', 'output_size', 'num_params'),
            row_settings=['var_names']
        )

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  weight_decay=config['weight_decay'], 
                                  lr=config['lr'])
    log.debug(f"Optimizer: AdamW with weight decay: {config['weight_decay']} and learning rate: {config['lr']}")

    # For classification, always use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    log.debug("Using Cross Entropy Loss")

    # exit(0) # temporary exit to avoid running training during testing of the script

    # ==========================================================================================
    # ================================== LR SCHEDULER & EARLY STOPPING ==========================
    # Initialize LR scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stop_patience', 10),
        min_delta=config.get('early_stop_min_delta', 0.001),
        verbose=True
    )

    # ========================================================================================== 
    # ===================================== TRAINING LOOP ======================================
    log.info("Starting training loop...")

    # Initialize classes for saving best models
    save_best_model = SaveBestModel()
    # save_best_acc = SaveBestModelIOU()  # Repurpose for accuracy

    # Initialize lists to store training and validation loss and metrics
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    
    for epoch in range(config['epochs']):
        print(f"EPOCH: {epoch + 1}/{config['epochs']}")

        # Train for one epoch
        train_results = train(
            model,
            train_loader,
            optimizer,
            criterion,
            task='cls',
            logger=log,
            device=device,
            scheduler=scheduler  # Pass scheduler to step per epoch
        )
        # For cls: train_results = (train_loss, train_acc)
        train_epoch_loss, train_epoch_acc = train_results

        # Validate for one epoch
        val_results = validate(
            model,
            val_loader,
            criterion,
            task='cls',
            logger=log,
            phase='val',
            device=device
        )
        # For cls: val_results = (val_loss, val_acc)
        valid_epoch_loss, valid_epoch_acc = val_results

        # Append the metrics to the lists
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)

        # Step the LR scheduler (for epoch-based schedulers)
        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) or \
           isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_epoch_loss)  # Step with validation loss

        # Check early stopping
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Save the best models based on loss
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )

        # Log current LR for transparency
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f}, "
            f"Train Epoch Acc: {train_epoch_acc:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f}, "
            f"Valid Epoch Acc: {valid_epoch_acc:.4f}"
        )

        print('-' * 50)

    log.info("Training loop completed.")
    # ==================================================================================


    # =================================== SAVE MODELS ==================================
    log.info("Saving plots and final model ...")

    # Save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss, train_acc, valid_acc, out_dir, task='cls')

    # Save final model
    save_model(config['epochs'], model, optimizer, criterion, out_dir, name='final_model')
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
            'test_f1_score': None
        }
    else:
        checkpoint = torch.load(os.path.join(out_dir, 'best_model_loss.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Compute predictions and labels for confusion matrix
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                pixel_values, labels = data[0].to(device), data[1].to(device)
                outputs = model(pixel_values)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Print confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        # Save confusion matrix image
        plot_confusion_matrix(cm, classes, out_dir, title='Brain Planes Classification Confusion Matrix')

        # Generate classification report
        class_report = classification_report(all_labels, all_preds, target_names=classes)
        print("\nClassification Report:")
        print(class_report)

        # Save classification report to file
        with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
            f.write("Brain Planes Classification Report\n")
            f.write("="*50 + "\n\n")
            f.write(class_report)

        # Compute F1-score
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted average

        # Compute test loss and accuracy
        test_results = validate(
            model,
            test_loader,
            criterion,
            task='cls',
            logger=log,
            phase='test',
            device=device
        )
        test_loss, test_acc = test_results

        # Save final metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': valid_loss,
            'val_acc': valid_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1_score': float(f1)
        }
    
    log.info("Saving metrics to JSON file...")
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    log.info("Test phase completed.")
    # ================================================================================== 
    
    
    log.info("Task completed successfully.")
