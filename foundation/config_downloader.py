import ssl
import urllib.request
import os
import argparse

def download_dinov2_config(
    backbone_size="small",  # "small", "base", "large", "giant"
    head_dataset="voc2012",  # "ade20k", "voc2012"
    head_type="ms",  # "ms", "linear"
    output_dir="configs"
):
    """
    Download the DINOv2 configuration file for the specified parameters.

    Args:
        backbone_size (str): Backbone size ("small", "base", "large", "giant").
        head_dataset (str): Head dataset ("ade20k", "voc2012").
        head_type (str): Head type ("ms", "linear").
        output_dir (str): Output directory for the configuration file.

    Returns:
        str: Path to the downloaded configuration file.

    Raises:
        ValueError: If an invalid parameter is provided.
        URLError: If the download fails.
    """
    dinov2_base_url = "https://dl.fbaipublicfiles.com/dinov2"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    if backbone_size not in backbone_archs:
        raise ValueError(f"Invalid backbone_size '{backbone_size}'. Must be one of {list(backbone_archs.keys())}.")
    if head_dataset not in {"ade20k", "voc2012"}:
        raise ValueError(f"Invalid head_dataset '{head_dataset}'. Must be 'ade20k' or 'voc2012'.")
    if head_type not in {"ms", "linear"}:
        raise ValueError(f"Invalid head_type '{head_type}'. Must be 'ms' or 'linear'.")

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    filename = f"{backbone_name}_{head_dataset}_{head_type}_config.py"
    # cropped_filename = f"{backbone_name}_{head_type}_cfg.py"
    cropped_filename = "dinov2_backbone_cfg.py"
    output_path = os.path.join(output_dir, cropped_filename)
    head_config_url = f"{dinov2_base_url}/{backbone_name}/{filename}"

    os.makedirs(output_dir, exist_ok=True)
    
    ssl_context = ssl._create_unverified_context()  # Bypass SSL verification for the download
    try:
        with urllib.request.urlopen(head_config_url, context=ssl_context) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Failed to download config from {head_config_url}: {e}")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DINOv2 configuration file.")
    parser.add_argument("--backbone", type=str, default="small", choices=["small", "base", "large", "giant"], help="Backbone size")
    parser.add_argument("--head_dataset", type=str, default="voc2012", choices=["ade20k", "voc2012"], help="Head dataset")
    parser.add_argument("--head_type", type=str, default="ms", choices=["ms", "linear"], help="Head type")
    parser.add_argument("--output_dir", type=str, default="configs", help="Output directory")

    args = parser.parse_args()

    try:
        output_path = download_dinov2_config(
            backbone_size=args.backbone,
            head_dataset=args.head_dataset,
            head_type=args.head_type,
            output_dir=args.output_dir
        )
        print(f"Config downloaded to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")