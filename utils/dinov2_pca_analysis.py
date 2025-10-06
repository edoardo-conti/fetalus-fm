import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.manifold import TSNE

import torch
import torchvision.transforms as T

# Import del dataset custom
from datasets.fpdb import VD_FPDB

# ================================================================================
# Configuration
# ================================================================================
IMG_SIZE = 224
OUTPUT_FOLDER = Path("/leonardo_work/IscrC_FoSAM-X/fetalus-fm/dinov2_pca_brain_analysis")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Process images in batches for efficiency
DF_SAMPLES_LIMIT = 0 # 0 = no limit, otherwise max samples per brain_plane
N_SAMPLES_PER_PLANE = 100 # Number of sample visualizations to save per brain plane

# Ensure output folder exists
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

assert IMG_SIZE % 14 == 0, "The image size must be exactly divisible by 14"

print(f"Using device: {DEVICE}")

# ================================================================================
# Load DINOv2 model
# ================================================================================
print("Loading DINOv2 model...")
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14.to(DEVICE)
dinov2_vitl14.eval()

# Define transformation
transform = T.Compose([
    T.ToTensor(),
    T.Resize(IMG_SIZE + int(IMG_SIZE * 0.01) * 10),
    T.CenterCrop(IMG_SIZE),
    T.Normalize([0.5], [0.5]),
])

# ================================================================================
# Helper Functions
# ================================================================================
def extract_features_batch(images: torch.Tensor) -> np.ndarray:
    """Extract DINOv2 features for a batch of images."""
    with torch.no_grad():
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        embeddings = dinov2_vitl14.forward_features(images.to(DEVICE))
        x_norm_patchtokens = embeddings["x_norm_patchtokens"].cpu().numpy()
    
    return x_norm_patchtokens

def compute_pca_features(x_norm_patchtokens: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Compute PCA features from patch tokens."""
    patch_h = patch_w = IMG_SIZE // 14
    batch_size = x_norm_patchtokens.shape[0]
    
    # Reshape to (batch_size, num_patches, features)
    x_norm_patches = x_norm_patchtokens.reshape(batch_size, patch_h * patch_w, -1)
    
    all_pca_features = []
    for i in range(batch_size):
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(x_norm_patches[i])
        pca_features = minmax_scale(pca_features)
        all_pca_features.append(pca_features)
    
    return np.array(all_pca_features)

def compute_global_features(pca_features: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute global features from PCA features for each image."""
    batch_size = pca_features.shape[0]
    n_components = pca_features.shape[2]
    
    features = {
        'mean': np.zeros((batch_size, n_components)),
        'std': np.zeros((batch_size, n_components)),
        'max': np.zeros((batch_size, n_components)),
        'min': np.zeros((batch_size, n_components))
    }
    
    for i in range(batch_size):
        for j in range(n_components):
            component = pca_features[i, :, j]
            features['mean'][i, j] = np.mean(component)
            features['std'][i, j] = np.std(component)
            features['max'][i, j] = np.max(component)
            features['min'][i, j] = np.min(component)
    
    return features

# ================================================================================
# Main Analysis
# ================================================================================
def main():
    # Initialize dataset
    print("\nInitializing FPDB dataset...")
    root_path = Path("/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/FPDB")  # Adjust this path
    
    # Load all splits
    datasets = {
        'train': VD_FPDB(root=root_path, split='train'),
        'val': VD_FPDB(root=root_path, split='val'),
        'test': VD_FPDB(root=root_path, split='test')
    }
    
    # Collect all brain images (not NaB)
    all_brain_data = []
    all_features = []
    all_pca_components = []
    
    print("\nCollecting brain plane images...")
    for split_name, dataset in datasets.items():
        df = dataset.data
        # Filter for brain planes (not NaB)
        brain_df = df[(df['class'] == 'Brain') & (df['brain_plane'] != 'NaB')]
        
        print(f"\n{split_name} split: {len(brain_df)} brain plane images")
        print(f"Brain plane distribution:\n{brain_df['brain_plane'].value_counts()}")
        
        if DF_SAMPLES_LIMIT > 0:
            # Limit to maximum 50 images per brain_plane
            brain_df_limited = brain_df.groupby('brain_plane').apply(
                lambda x: x.sample(n=min(DF_SAMPLES_LIMIT, len(x)), random_state=42)
            ).reset_index(drop=True)
            
            print(f"After limiting: {len(brain_df_limited)} brain plane images")
            print(f"Limited brain plane distribution:\n{brain_df_limited['brain_plane'].value_counts()}")
        
        # Process images in batches
        batch_images = []
        batch_info = []
        
        if DF_SAMPLES_LIMIT > 0:
            brain_df_to_process = brain_df_limited
        else:
            brain_df_to_process = brain_df

        for idx, row in tqdm(brain_df_to_process.iterrows(), total=len(brain_df_to_process), 
                            desc=f"Processing {split_name}"):
            # Load and transform image
            img_path = root_path / row['image_path']
            if not img_path.exists():
                print(f"Warning: Image not found {img_path}")
                continue
            
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image_tensor = transform(image)
            
            batch_images.append(image_tensor)
            batch_info.append({
                'brain_plane': row['brain_plane'],
                'patient_id': row['patient_id'],
                'image_path': row['image_path'],
                'split': split_name
            })
            
            # Process batch when full or at the end
            if len(batch_images) == BATCH_SIZE or idx == brain_df_to_process.index[-1]:
                batch_tensor = torch.stack(batch_images)
                
                # Extract features
                patch_tokens = extract_features_batch(batch_tensor)
                
                # Compute PCA features
                pca_features = compute_pca_features(patch_tokens, n_components=3)
                
                # Compute global features
                global_features = compute_global_features(pca_features)
                
                # Store results
                for i in range(len(batch_images)):
                    info = batch_info[i]
                    info.update({
                        'pca_mean': global_features['mean'][i],
                        'pca_std': global_features['std'][i],
                        'pca_max': global_features['max'][i],
                        'pca_min': global_features['min'][i]
                    })
                    all_brain_data.append(info)
                    all_features.append(np.concatenate([
                        global_features['mean'][i],
                        global_features['std'][i]
                    ]))
                    all_pca_components.append(pca_features[i])
                
                # Clear batch
                batch_images = []
                batch_info = []
    
    print(f"\nTotal brain plane images processed: {len(all_brain_data)}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_brain_data)
    features_array = np.array(all_features)
    
    # ================================================================================
    # Visualization
    # ================================================================================
    print("\nCreating visualizations...")
    
    # Color mapping for brain planes
    brain_planes = df_results['brain_plane'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(brain_planes)))
    color_map = dict(zip(brain_planes, colors))
    
    # 1. PCA Feature Distribution (2D projection of the 6D feature space)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot first 3 PCA components against each other
    component_pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['PC1 Mean vs PC2 Mean', 'PC1 Mean vs PC3 Mean', 'PC2 Mean vs PC3 Mean']
    
    for idx, (i, j) in enumerate(component_pairs):
        ax = axes[0, idx]
        for plane in brain_planes:
            mask = df_results['brain_plane'] == plane
            ax.scatter(features_array[mask, i], features_array[mask, j], 
                      label=plane, alpha=0.6, c=[color_map[plane]], s=30)
        ax.set_xlabel(f'PC{i+1} Mean')
        ax.set_ylabel(f'PC{j+1} Mean')
        ax.set_title(titles[idx])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot with std components
    std_pairs = [(3, 4), (3, 5), (4, 5)]
    titles_std = ['PC1 Std vs PC2 Std', 'PC1 Std vs PC3 Std', 'PC2 Std vs PC3 Std']
    
    for idx, (i, j) in enumerate(std_pairs):
        ax = axes[1, idx]
        for plane in brain_planes:
            mask = df_results['brain_plane'] == plane
            ax.scatter(features_array[mask, i], features_array[mask, j], 
                      label=plane, alpha=0.6, c=[color_map[plane]], s=30)
        ax.set_xlabel(f'PC{i-2} Std')
        ax.set_ylabel(f'PC{j-2} Std')
        ax.set_title(titles_std[idx])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('PCA Features Distribution by Brain Plane', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'pca_features_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Dimensionality Reduction Visualizations
    print("Computing dimensionality reductions...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # PCA 2D
    pca_2d = PCA(n_components=2)
    features_pca_2d = pca_2d.fit_transform(features_scaled)
    
    ax = axes[0]
    for plane in brain_planes:
        mask = df_results['brain_plane'] == plane
        ax.scatter(features_pca_2d[mask, 0], features_pca_2d[mask, 1], 
                  label=plane, alpha=0.6, c=[color_map[plane]], s=30)
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA Projection (2D)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_array)-1))
    features_tsne = tsne.fit_transform(features_scaled)
    
    ax = axes[1]
    for plane in brain_planes:
        mask = df_results['brain_plane'] == plane
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                  label=plane, alpha=0.6, c=[color_map[plane]], s=30)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # UMAP
    # reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(features_array)-1))
    # features_umap = reducer.fit_transform(features_scaled)
    
    # ax = axes[2]
    # for plane in brain_planes:
    #     mask = df_results['brain_plane'] == plane
    #     ax.scatter(features_umap[mask, 0], features_umap[mask, 1], 
    #               label=plane, alpha=0.6, c=[color_map[plane]], s=30)
    # ax.set_xlabel('UMAP 1')
    # ax.set_ylabel('UMAP 2')
    # ax.set_title('UMAP Projection')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dimensionality Reduction of PCA Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'dimensionality_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions by brain plane
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i in range(3):
        # Mean distributions
        ax = axes[0, i]
        for plane in brain_planes:
            mask = df_results['brain_plane'] == plane
            ax.hist(features_array[mask, i], alpha=0.5, label=plane, bins=20, 
                   color=color_map[plane], density=True)
        ax.set_xlabel(f'PC{i+1} Mean')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of PC{i+1} Mean')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Std distributions
        ax = axes[1, i]
        for plane in brain_planes:
            mask = df_results['brain_plane'] == plane
            ax.hist(features_array[mask, i+3], alpha=0.5, label=plane, bins=20, 
                   color=color_map[plane], density=True)
        ax.set_xlabel(f'PC{i+1} Std')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of PC{i+1} Std')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions by Brain Plane', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for plane in brain_planes:
        mask = df_results['brain_plane'] == plane
        ax.scatter(features_array[mask, 0], 
                  features_array[mask, 1], 
                  features_array[mask, 2],
                  label=plane, alpha=0.6, c=[color_map[plane]], s=30)
    
    ax.set_xlabel('PC1 Mean')
    ax.set_ylabel('PC2 Mean')
    ax.set_zlabel('PC3 Mean')
    ax.set_title('3D PCA Features Space')
    ax.legend()
    plt.savefig(OUTPUT_FOLDER / 'pca_features_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Save sample visualizations for each brain plane
    print("\nSaving sample visualizations...")
    samples_folder = OUTPUT_FOLDER / 'samples'
    samples_folder.mkdir(exist_ok=True)
    
    for plane in brain_planes:
        # Create a subfolder for each plane
        plane_folder = samples_folder / plane
        plane_folder.mkdir(exist_ok=True)

        plane_data = df_results[df_results['brain_plane'] == plane]
        
        # Take up to N samples
        num_samples_to_save = min(N_SAMPLES_PER_PLANE, len(plane_data))
        
        if num_samples_to_save > 0:
            print(f"  Saving {num_samples_to_save} samples for plane: {plane}")

        for i in range(num_samples_to_save):
            sample = plane_data.iloc[i]
            img_path = root_path / sample['image_path']
            
            if img_path.exists():
                # Load and process image
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0)
                
                # Extract features
                patch_tokens = extract_features_batch(image_tensor)
                pca_features = compute_pca_features(patch_tokens, n_components=3)[0]
                
                # Reshape for visualization
                patch_h = patch_w = IMG_SIZE // 14
                pca_rgb = pca_features.reshape(patch_h, patch_w, 3)
                
                # Create visualization
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                # Original image (resized for consistency)
                axes[0].imshow(image.resize((IMG_SIZE, IMG_SIZE)), cmap='gray')
                axes[0].set_title(f'Image ({plane})')
                axes[0].axis('off')
                
                # Individual PCA components
                for j in range(3):
                    axes[j+1].imshow(pca_features[:, j].reshape(patch_h, patch_w), 
                                    cmap='viridis')
                    axes[j+1].set_title(f'PC{j+1}')
                    axes[j+1].axis('off')
                
                # RGB visualization
                axes[4].imshow(pca_rgb)
                axes[4].set_title('PCA as RGB')
                axes[4].axis('off')
                
                plt.suptitle(f'Brain Plane: {plane} - Sample {i+1}', fontsize=14)
                plt.tight_layout()
                
                # Use original image name for the saved file to avoid overwrites and add context
                save_filename = f"{img_path.stem}_visualization.png"
                plt.savefig(plane_folder / save_filename, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print(f"Warning: Sample image not found {img_path}")
    
    # Save results to CSV
    df_results.to_csv(OUTPUT_FOLDER / 'brain_planes_pca_features.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(df_results)}")
    print(f"\nBrain plane distribution:")
    print(df_results['brain_plane'].value_counts())
    print(f"\nSplit distribution:")
    print(df_results['split'].value_counts())
    
    # Compute separability metrics
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, classification_report
    
    # Silhouette score
    labels = df_results['brain_plane'].values
    silhouette = silhouette_score(features_scaled, labels)
    print(f"\nSilhouette Score: {silhouette:.3f}")
    
    # LDA for separability
    lda = LinearDiscriminantAnalysis()
    lda.fit(features_scaled, labels)
    lda_score = lda.score(features_scaled, labels)
    print(f"LDA Classification Accuracy: {lda_score:.3f}")
    
    # ================================================================================
    # Brain Plane Classification Analysis (using raw DINOv2 features)
    # ================================================================================
    print("\n" + "="*60)
    print("BRAIN PLANE CLASSIFICATION ANALYSIS (RAW DINOv2 FEATURES)")
    print("="*60)
    
    # Collect raw DINOv2 features instead of PCA features
    print("\nCollecting raw DINOv2 features...")
    all_raw_features = []
    all_brain_data_raw = []
    
    for split_name, dataset in datasets.items():
        df = dataset.data
        # Filter for brain planes (not NaB)
        brain_df = df[(df['class'] == 'Brain') & (df['brain_plane'] != 'NaB')]
        
        if DF_SAMPLES_LIMIT > 0:
            brain_df = brain_df.groupby('brain_plane').apply(
                lambda x: x.sample(n=min(DF_SAMPLES_LIMIT, len(x)), random_state=42)
            ).reset_index(drop=True)
        
        # Process images in batches
        batch_images = []
        batch_info = []
        
        for idx, row in tqdm(brain_df.iterrows(), total=len(brain_df), 
                            desc=f"Extracting raw features {split_name}"):
            # Load and transform image
            img_path = root_path / row['image_path']
            if not img_path.exists():
                print(f"Warning: Image not found {img_path}")
                continue
            
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image_tensor = transform(image)
            
            batch_images.append(image_tensor)
            batch_info.append({
                'brain_plane': row['brain_plane'],
                'patient_id': row['patient_id'],
                'image_path': row['image_path'],
                'split': split_name
            })
            
            # Process batch when full or at the end
            if len(batch_images) == BATCH_SIZE or idx == brain_df.index[-1]:
                batch_tensor = torch.stack(batch_images)
                
                # Extract raw DINOv2 features (cls token)
                with torch.no_grad():
                    if batch_tensor.shape[1] == 1:
                        batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
                    
                    embeddings = dinov2_vitl14.forward_features(batch_tensor.to(DEVICE))
                    # Use cls token as global feature representation
                    cls_features = embeddings["x_norm_clstoken"].cpu().numpy()
                
                # Store results
                for i in range(len(batch_images)):
                    info = batch_info[i]
                    all_brain_data_raw.append(info)
                    all_raw_features.append(cls_features[i])
                
                # Clear batch
                batch_images = []
                batch_info = []
    
    # Convert to arrays
    raw_features_array = np.array(all_raw_features)
    df_raw_results = pd.DataFrame(all_brain_data_raw)
    
    print(f"\nTotal brain plane images with raw features: {len(df_raw_results)}")
    
    # Prepare data for classification using original dataset splits
    train_mask = df_raw_results['split'] == 'train'
    test_mask = df_raw_results['split'] == 'test'
    
    X_train_raw = raw_features_array[train_mask]
    y_train_raw = df_raw_results['brain_plane'].values[train_mask]
    X_test_raw = raw_features_array[test_mask]
    y_test_raw = df_raw_results['brain_plane'].values[test_mask]
    
    print(f"Training set size: {len(X_train_raw)} images")
    print(f"Test set size: {len(X_test_raw)} images")
    print(f"Training set distribution: {pd.Series(y_train_raw).value_counts().to_dict()}")
    print(f"Test set distribution: {pd.Series(y_test_raw).value_counts().to_dict()}")
    
    # Standardize features
    scaler_raw = StandardScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_scaled = scaler_raw.transform(X_test_raw)
    
    # Use Linear Discriminant Analysis as the base classifier
    clf_raw = LinearDiscriminantAnalysis()
    clf_name_raw = "Linear Discriminant Analysis (Raw DINOv2 Features)"
    
    # Train the classifier
    print(f"\nTraining {clf_name_raw} on training split...")
    clf_raw.fit(X_train_scaled, y_train_raw)
    
    # Evaluate on test set
    print("Evaluating on test split...")
    y_pred_raw = clf_raw.predict(X_test_scaled)
    accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
    
    print(f"\n{clf_name_raw} Results:")
    print(f"Test Accuracy: {accuracy_raw:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_raw, y_pred_raw, target_names=brain_planes))
    
    # Confusion matrix
    cm_raw = confusion_matrix(y_test_raw, y_pred_raw, labels=brain_planes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', 
                xticklabels=brain_planes, yticklabels=brain_planes,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {clf_name_raw}\n(Test Accuracy: {accuracy_raw:.3f})')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'confusion_matrix_raw_features_test_split.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_raw_features_test_split.png'")
    
    # Save detailed metrics to file
    metrics_file_raw = OUTPUT_FOLDER / 'test_set_metrics_raw_features.txt'
    with open(metrics_file_raw, 'w') as f:
        f.write("BRAIN PLANE CLASSIFICATION RESULTS (RAW DINOv2 FEATURES)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classifier: {clf_name_raw}\n")
        f.write(f"Training set size: {len(X_train_raw)}\n")
        f.write(f"Test set size: {len(X_test_raw)}\n")
        f.write(f"Test Accuracy: {accuracy_raw:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test_raw, y_pred_raw, target_names=brain_planes))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm_raw))
    
    print(f"Detailed metrics saved to 'test_set_metrics_raw_features.txt'")
    
    # Cross-validation on training set for additional insight
    print(f"\nCross-validation on training set:")
    cv_raw = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_raw = cross_val_score(clf_raw, X_train_scaled, y_train_raw, cv=cv_raw, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores_raw.mean():.3f} ± {cv_scores_raw.std():.3f}")
    
    print(f"\nResults saved in: {OUTPUT_FOLDER}")
    print("="*60)
    
    # Final answer to the research question
    print("\n" + "="*60)
    print("CONCLUSION: CAN DINOv2 DISTINGUISH BRAIN PLANES? (RAW FEATURES)")
    print("="*60)
    print(f"Based on the analysis of {len(df_raw_results)} brain plane images:")
    print(f"- Test accuracy on original dataset splits: {accuracy_raw:.3f}")
    print(f"- Cross-validation accuracy on training set: {cv_scores_raw.mean():.3f} ± {cv_scores_raw.std():.3f}")
    print(f"- Number of distinct brain planes: {len(brain_planes)}")
    print(f"- Feature dimension: {raw_features_array.shape[1]}")
    print("="*60)

if __name__ == "__main__":
    main()
