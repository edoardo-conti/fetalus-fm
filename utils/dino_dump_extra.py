import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from dinov3.data.datasets import FetalUS

for split in FetalUS.Split:
    dataset = FetalUS(split=split, root="/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/UNSUPERVISED", extra="/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/UNSUPERVISED")
    dataset.dump_extra()
