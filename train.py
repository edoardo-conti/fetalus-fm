import os
import argparse
import json
import warnings
import shutil
import torch
import torch.nn as nn

from datetime import datetime
from torchinfo import summary
from torch.utils.data import DataLoader
from augmentations import geometric_transforms, color_transforms
from datasets.unified import UnifiedFetalDataset
from foundation.model import Dinov2Segmentation
from utils import IMAGE_SIZE, SEED, FUS_STRUCTURES, FUS_STRUCTURES_COLORS, SaveBestModel, SaveBestModelIOU, save_model, save_plots
from engine import train, validate

# Suppress warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# Set random seed for reproducibility
torch.manual_seed(SEED)

# Set random seed for CUDA and MPS if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="root of the project")
parser.add_argument("--datasets", type=str, required=True, help="path to the datasets")
parser.add_argument("--exp-id", type=int, help="index of experiment to run from config")
parser.add_argument('--fine-tune', dest='fine_tune', action='store_true', help='whether to fine tune the backbone or not')
parser.add_argument('--debug', action='store_true', help='debug mode, reduces dataset size for faster testing')
args = parser.parse_args()

# print(f"Arguments: {args}")

if __name__ == '__main__':
    
    # Load experiment config
    with open(os.path.join(args.root, 'configs/experiments.json')) as f:
        experiments = json.load(f)
    
    # Handle exp-id argument robustly: make it optional and check validity
    if args.exp_id is not None:
        if args.exp_id < 0 or args.exp_id >= len(experiments):
            raise ValueError(f"exp-id {args.exp_id} is out of range (0-{len(experiments)-1})")
        config = experiments[args.exp_id]
    else:
        # If exp-id is not provided, use the first experiment as default
        config = experiments[0]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #timestamp = datetime.now().strftime("%Y%m%d")
    exp_name = config['datasets'][0] if len(config['datasets']) == 1 else ''.join([d[0].lower() for d in config['datasets']])
    out_dir = os.path.join(args.root, 'outputs', f"{exp_name}_{timestamp}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    val_preds_dir = os.path.join(out_dir, 'val_preds')
    test_preds_dir = os.path.join(out_dir, 'test_preds')
    os.makedirs(val_preds_dir, exist_ok=True)
    os.makedirs(test_preds_dir, exist_ok=True)
    
    # Save config copy
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment name: {exp_name}")
    print(f"Output directory: {out_dir}\n")

    # ================================================================================== 
    # ==================================== DATASET =====================================
    # ================================================================================== 
    train_geo_tfms = geometric_transforms(image_size=IMAGE_SIZE)
    train_color_tfms = color_transforms()
    
    fus_train = UnifiedFetalDataset(
        root=args.root,
        data_path=args.datasets,
        datasets=config['datasets'],
        split='train',
        supervised=True,
        target_size=IMAGE_SIZE,
        augmentations=(train_geo_tfms, train_color_tfms),
    )
    fus_val = UnifiedFetalDataset(
        root=args.root,
        data_path=args.datasets,
        datasets=fus_train.datasets,
        split='val',
        supervised=True,
        target_size=IMAGE_SIZE,
    )
    fus_test = UnifiedFetalDataset(
        root=args.root,
        data_path=args.datasets,
        datasets=fus_train.datasets,
        split='test',
        supervised=True,
        target_size=IMAGE_SIZE,
    )
    
    # DataLoaders
    if args.debug:
        # Reduce dataset size for debugging
        fus_train = torch.utils.data.Subset(fus_train, range(min(2, len(fus_train))))
        fus_val = torch.utils.data.Subset(fus_val, range(min(1, len(fus_val))))
        fus_test = torch.utils.data.Subset(fus_test, range(min(1, len(fus_test))))
    
    train_loader = DataLoader(fus_train, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(fus_val, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(fus_test, batch_size=config['batch_size'], shuffle=False)

    print(f"Train dataset size: {len(fus_train)}")
    print(f"Validation dataset size: {len(fus_val)}")
    print(f"Test dataset size: {len(fus_test)}")
    
    # classes
    classes = FUS_STRUCTURES
    num_classes = len(classes)
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}\n")

    # ================================================================================== 
    # ===================================== MODEL ======================================
    # ================================================================================== 
    # Set device based on availability
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    # device = "cpu" # for testing purposes because it doesn't work on MPS
    print(f"Using device: {device}")

    dino_backbone = config['dino']['backbone']
    model = Dinov2Segmentation(nc=num_classes, backbone=dino_backbone, fine_tune=args.fine_tune, device=device)
    model.decode_head.conv_seg = nn.Conv2d(1536, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.to(device)

    summary(
        model, 
        (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  weight_decay=config['weight_decay'], 
                                  lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    

    # ================================================================================== 
    # ================================= TRAINING LOOP ==================================
    # ================================================================================== 
    # Initialize some classes
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()

    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    
    for epoch in range (config['epochs']):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            classes
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            val_loader,
            device,
            criterion,
            classes,
            FUS_STRUCTURES_COLORS,
            epoch,
            save_dir=val_preds_dir
        )

        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

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
    # =================================== TEST PHASE ===================================
    # ================================================================================== 
    # Skip test phase if test set is empty
    if len(fus_test) == 0:
        print("Test set is empty, skipping test phase")
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
        checkpoint = torch.load(os.path.join(out_dir, 'final_model.pth'), weights_only=False)
        model.to(device)
        model.eval()
        
        test_loss, test_pixacc, test_miou = validate(
            model,
            test_loader,
            device,
            criterion,
            classes,
            FUS_STRUCTURES_COLORS,
            epoch=0,
            save_dir=test_preds_dir,
            phase='test'
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

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print('TRAINING COMPLETE')
