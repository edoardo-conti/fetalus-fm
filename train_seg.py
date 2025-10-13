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

from datetime import datetime
from torchinfo import summary
from torch.utils.data import DataLoader
from foundation.model import DINOSegmentator
from utils.utils import DEFAULT_IMAGE_SIZE, SEED, FUS_STRUCTS, SaveBestModel, SaveBestModelIOU, save_model, save_plots
from engine import train, validate
from losses import DiceLoss, CEDiceLoss

from datahandler import get_fetalus_dataloaders


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

    # Create output directory of the training experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['datasets'][0] if len(config['datasets']) == 1 else ''.join([d[0].lower() for d in config['datasets']])
    out_dir = os.path.join(args.root, 'outputs', f"{exp_name}_{timestamp}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save config copy
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

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
    fus_train, fus_val, fus_test = get_fetalus_dataloaders(
        root=args.root,
        data_path=args.datasets,
        datasets=config['datasets'],
        image_size=config['image_size'],
        supervised=True,
        debug=args.debug)
    
    # Create DataLoaders 
    train_loader = DataLoader(fus_train, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(fus_val, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(fus_test, batch_size=config['batch_size'], shuffle=False)

    # classes
    classes = FUS_STRUCTS
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
    model = DINOSegmentator(nc=num_classes, 
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

    if config['loss'] == 'DICE':
        criterion = DiceLoss()
        log.debug("Using Dice Loss")
    elif config['loss'] == 'CEDICE':
        criterion = CEDiceLoss()
        log.debug("Using CEDice Loss (Cross Entropy + Dice Loss)")
    else:
        # Default to Cross entropy loss (CE)
        criterion = nn.CrossEntropyLoss()    
        log.debug("Using Cross Entropy Loss")

    # ========================================================================================== 
    # ===================================== TRAINING LOOP ======================================
    log.info("Starting training loop...")

    # Initialize classes for saving best models
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    
    # Initialize lists to store training and validation loss and metrics
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    
    for epoch in range (config['epochs']):
        print(f"EPOCH: {epoch + 1}/{config['epochs']}")

        # train and validate the model for one epoch
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_loader,
            optimizer,
            criterion,
            num_classes,
            logger=log,
            device=device,
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            val_loader,
            criterion,
            num_classes,
            logger=log,
            phase='val',
            device=device
        )

        # Append the metrics to the lists
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)
        
        # Save the best models based on loss and mIOU
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )
        
        print('-' * 50)

    log.info("Training loop completed.")
    # ==================================================================================


    # =================================== SAVE MODELS ==================================
    log.info("Saving plots and final model ...")

    # Save the loss and accuracy plots
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )

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
            'train_acc': train_pix_acc,
            'train_iou': train_miou,
            'val_loss': valid_loss,
            'val_acc': valid_pix_acc,
            'val_iou': valid_miou,
            'test_loss': None,
            'test_acc': None,
            'test_iou': None
        }
    else:
        checkpoint = torch.load(os.path.join(out_dir, 'best_model_iou.pth'), weights_only=False)
        model.eval()
        
        test_loss, test_pixacc, test_miou = validate(
            model,
            test_loader,
            criterion,
            num_classes,
            logger=log,
            phase='test',
            device=device
        )
        
        # Save final metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_pix_acc,
            'train_iou': train_miou,
            'val_loss': valid_loss,
            'val_acc': valid_pix_acc,
            'val_iou': valid_miou,
            'test_loss': test_loss,
            'test_acc': test_pixacc,
            'test_iou': test_miou
        }

    log.info("Saving metrics to JSON file...")
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    log.info("Test phase completed.")
    # ================================================================================== 
    
    
    log.info("Task completed successfully.")
