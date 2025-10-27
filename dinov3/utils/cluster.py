# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterType(Enum):
    CW = "cw"
    LEONARDO = "leonardo"


def _guess_cluster_type() -> ClusterType:
    # Detect cluster type based on environment
    hostname = os.environ.get("HOSTNAME", "")
    cluster_name = os.environ.get("SLURM_CLUSTER_NAME", "")

    if "leonardo" in hostname.lower() or "leonardo" in cluster_name.lower():
        return ClusterType.LEONARDO
    else:
        return ClusterType.CW


def get_cluster_type(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_slurm_account(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None
    return {
        ClusterType.CW: "fair_amaia_cw_explore",
        ClusterType.LEONARDO: "IscrC_FoSAM-X",
    }.get(cluster_type)


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    CHECKPOINT_DIRNAMES = {
        ClusterType.CW: "",
        ClusterType.LEONARDO: "leonardo_work/IscrC_FoSAM-X/fetalus-fm/dinov3_checkpoints",
    }
    return Path("/") / CHECKPOINT_DIRNAMES[cluster_type]


def get_user_checkpoint_path(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    username = os.environ.get("USER")
    assert username is not None
    return checkpoint_path / username


def get_slurm_qos(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    return {
        ClusterType.CW: "explore",
        ClusterType.LEONARDO: None,  # Disable QoS for LeonardoB
    }.get(cluster_type)


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    SLURM_PARTITIONS = {
        ClusterType.CW: "learn",
        ClusterType.LEONARDO: "boost_usr_prod",
    }
    return SLURM_PARTITIONS[cluster_type]


def get_slurm_executor_parameters(
    nodes: int,
    num_gpus_per_node: int,
    cluster_type: Optional[ClusterType] = None,
    **kwargs,
) -> Dict[str, Any]:
    # Determine cluster type
    cluster_type = get_cluster_type(cluster_type)

    if cluster_type == ClusterType.LEONARDO:
        # LeonardoB boost_usr_prod node specifications
        # 128 cores total, 4 GPUs per node -> ~32 CPUs per GPU for optimal data loading
        params = {
            # "mem_gb": 480,  # 480GB per node as per DINOv3_TRAINING.sh
            # "timeout_min": 60,  # 1 hour as per DINOv3_TRAINING.sh for robust resource usage
            # "gpus_per_node": num_gpus_per_node,
            # "tasks_per_node": 1, 
            "cpus_per_task": 32, # 16 se ntask=1
            "nodes": nodes,
            "slurm_account": get_slurm_account(cluster_type),
            "slurm_partition": get_slurm_partition(cluster_type),
            "slurm_additional_parameters": {
                "time": "23:30:00",
                "ntasks": 1, # num_gpus_per_node
                "gres": f"gpu:a100:1" # f"gpu:a100:{num_gpus_per_node}
            },
            "slurm_setup": [
                "module load profile/base",
                "module load profile/archive",
                "module load python/3.10.8--gcc--8.5.0",
                "module load cuda",
                "source /leonardo_work/IscrC_FoSAM-X/fetalus-fm/.my_dinov3_env/bin/activate",
                "cd /leonardo_work/IscrC_FoSAM-X/fetalus-fm",
                "export PYTHONPATH=${PWD}:${PYTHONPATH}",
                "export TORCH_NCCL_TIMEOUT=7200000",
                'echo "===================================== System Info ====================================="',
                'echo "Node list: $SLURM_NODELIST"',
                'echo "CPUs per task: $SLURM_CPUS_PER_TASK"',
                'echo "GPUs: $CUDA_VISIBLE_DEVICES"',
                'echo "========================================================================================"'
            ],
        }
    else:
        # Default parameters (CW cluster)
        params = {
            "mem_gb": 0,  # Requests all memory on a node
            "gpus_per_node": num_gpus_per_node,
            "tasks_per_node": num_gpus_per_node,  # one task per GPU
            "cpus_per_task": 16,  # CW cluster default
            "nodes": nodes,
            "slurm_partition": get_slurm_partition(cluster_type),
        }

    # Set additional parameters / apply overrides
    params.update(kwargs)
    return params
