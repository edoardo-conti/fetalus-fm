import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


DEFAULT_IMAGE_SIZE = (224, 224) # (644,644) is the default image size used in DINOv2.
SEED = 42

DATASETS_CONFIGS = {
    'HC18': {
        'structures': ['BRAIN', 'CSP', 'LV'],
        'mask_color_map': {
            'BRAIN': [255, 0, 0],   # Red
            'CSP': [0, 255, 0],     # Green
            'LV': [0, 0, 255]       # Blue
        }
    },
    'FABD': {
        'structures': ['ARTERY', 'LIVER', 'STOMACH', 'VEIN'],
        'mask_color_map': {
            'ARTERY': [255, 0, 0],    # Red
            'LIVER': [0, 255, 0],     # Green
            'STOMACH': [0, 0, 255],   # Blue
            'VEIN': [255, 255, 0]     # Yellow
        }
    },
    'FPLR': {
        'structures': [],
        'mask_color_map': {}
    },
    'FPDB': {
        'structures': ['BRAIN', 'CSP', 'LV'],
        'mask_color_map': {
            'BRAIN': [255, 0, 0],       # Red
            'CSP': [0, 255, 0],         # Green
            'LV': [0, 0, 255]           # Blue
        }
    },
    'IPSFH': {
        'structures': ['PS', 'FH'],
        'mask_color_map': {
            'PS': [255, 0, 0],          # Red (Pubic Symphysis)
            'FH': [0, 255, 0]           # Green (Fetal Head)
        }
    },
    'ACSLC': {
        'structures': ['ABDOMEN'],
        'mask_color_map': {
            'ABDOMEN': [255, 0, 0]      # Red (Abdomen)
        }
    },
    'FECST': {
        'structures': ['S', 'VD', 'VS', 'AD', 'AS'],
        'mask_color_map': {
            'S': [255, 0, 0],           # Red (Stomach)      
            'VD': [0, 255, 0],          # Green (Right Ventricle)    
            'VS': [0, 0, 255],          # Blue (Left Ventricle)    
            'AD': [255, 255, 0],        # Yellow (Right Atrium)    
            'AS': [0, 255, 255]         # Cyan (Left Atrium)       
        }
    }
}


FPDB_BRAIN_PLANES = ['TT', # trans-thalamic
                     'TV', # trans-ventricular
                     'TC'] # trans-cerebellum


FUS_STRUCTS = [
    'BACKGROUND',
    'BRAIN',        # HC18, PlanesDB
    'CSP',          # HC18, PlanesDB 
    'LV',           # HC18, PlanesDB
    'ARTERY',       # Abdominal
    'LIVER',        # Abdominal
    'STOMACH',      # Abdominal
    'VEIN',         # Abdominal
    'FH',           # PSFH (Fetal Head)
    'PS',           # PSFH (Pubic Symphysis)
    'ABDOMEN',      # ACSLC (Abdomen)
    # 'S',            # FECST (Stomach)
    # 'VD',           # FECST (Right Ventricle)
    # 'VS',           # FECST (Left Ventricle)
    # 'AD',           # FECST (Right Atrium)
    # 'AS'            # FECST (Left Atrium)
]


FUS_STRUCTS_COLORS = {
    0: [0, 0, 0],       # BACKGROUND (Black)
    1: [255, 0, 0],     # BRAIN [HC18, FPDB](Red)
    2: [0, 255, 0],     # CSP [HC18, FPDB](Green)
    3: [0, 0, 255],     # LV [HC18, FPDB](Blue)
    4: [255, 255, 0],   # ARTERY [FADB](Yellow)
    5: [0, 255, 255],   # LIVER [FADB](Cyan)
    6: [255, 0, 255],   # STOMACH [FADB](Magenta)
    7: [128, 0, 0],     # VEIN [FADB](Maroon)
    8: [128, 128, 0],   # Fetal Head [IPSFH](Olive)
    9: [0, 128, 128],   # Pubic Symphysis [IPSFH](Teal)
    10: [255, 128, 0],  # Abdomen [ACSLC](Orange)
    # 10: [255, 64, 0],  # FECST Stomach (TODO: to fix)
    # 11: [128, 0, 255],  # FECST Right Ventricle (Purple)
    # 12: [0, 128, 255],  # FECST Left Ventricle (Light Blue)
    # 13: [255, 0, 128],  # FECST Right Atrium (Pink)
    # 14: [128, 255, 0]   # FECST Left Atrium (Light Green)
}

def mask2d_to_rgb(mask, structure_colors):
    """
    Convert segmentation mask to RGB using color mapping.
    
    Args:
        mask: Input segmentation mask (2D array)
        structure_colors: Dictionary mapping mask values to RGB colors
        
    Returns:
        RGB mask (3D array with shape [H, W, 3])
    """
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val, color in structure_colors.items():
        mask_rgb[mask == val] = color
    return mask_rgb


###########################################################################################
###################################### DEBUGGER CAFE ######################################
###########################################################################################
ALL_CLASSES = [
    'background',
    'brain',
    'csp',
    'lv',
]

# Define color mappings for each structure
LABEL_COLORS_LIST = [
    (0, 0, 0),      # Background
    [128, 0, 0],    # Brain (Maroon)
    [0, 128, 0],    # CSP (Green)
    [128, 128, 0],  # LV (Olive)
]

plt.style.use('ggplot')

def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g car=0, bus=1.
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask


def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir,
    label_colors_list
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """

    def _denormalize(x, mean=None, std=None):
        # x should be a Numpy array of shape [H, W, C] 
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        for t, m, s in zip(x, mean, std):
            t.mul_(s).add_(m)
        res = torch.clamp(t, 0, 1)
        res = res.squeeze(0).permute(1, 2, 0).numpy()
        return res

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast
    
    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = _denormalize(data[0].permute(1, 2, 0).cpu().numpy(), IMG_MEAN, IMG_STD)

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)

    for label_num in label_colors_list:
        index = seg_map == label_num
        red_map[index] = label_colors_list[label_num][0]
        green_map[index] = label_colors_list[label_num][1]
        blue_map[index] = label_colors_list[label_num][2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.uint8)  # Ensure uint8 type for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image * 255).astype(np.uint8)  # Convert to uint8 explicitly

    blended = cv2.addWeighted(image, alpha, rgb, beta, gamma)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", blended)


def save_model(epochs, model, optimizer, criterion, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'model': model
                }, os.path.join(out_dir, name+'.pth'))


def save_plots(
    train_acc, valid_acc,
    train_loss, valid_loss,
    train_miou, valid_miou,
    out_dir, task='seg'
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    
    # Third metric plots (mIOU for seg).
    if task == 'seg':
        ylabel = 'mIoU'
        label1 = 'train mIoU'
        label2 = 'validation mIoU'
        filename = 'miou.png'
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_miou, color='tab:blue', linestyle='-',
            label=label1
        )
        plt.plot(
            valid_miou, color='tab:red', linestyle='-',
            label=label2
        )
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(out_dir, filename))


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
    
    def __call__(self, current_valid_loss, epoch, model, out_dir, name='model'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for minimizing loss of epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))


class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's 
    IoU is higher than the previous highest, then save the
    model state.
    """
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou
    
    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for maximising IoU of epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            
###########################################################################################
##################################### \ DEBUGGER CAFE #####################################
###########################################################################################
