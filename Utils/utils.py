import yaml
from argparse import Namespace
import torch.distributed as dist
import torch

def load_args_from_file(config_path):
    """
    Load arguments from a YAML file and convert them into a Namespace.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Namespace: Arguments loaded as a Namespace object.
    """
    # Load YAML config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Convert dictionary to Namespace
    args = Namespace(**config)
    return args

def sequential_execution_by_rank(fn, *args, **kwargs):
    """Each rank executes `fn` in order of rank (0 to world_size-1)"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for r in range(world_size):
        dist.barrier()  # All ranks sync here
        if r == rank:
            fn(*args, **kwargs)
        dist.barrier()  # Ensure current rank is done before next