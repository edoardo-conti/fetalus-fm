from dinov3.data.datasets import FetalUS

for split in FetalUS.Split:
    dataset = FetalUS(split=split, root="/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/UNSUPERVISED", extra="/leonardo_scratch/fast/IscrC_FoSAM-X/datasets/UNSUPERVISED")
    dataset.dump_extra()