import os

import torch.distributed as dist


def distributed_backend() -> str:
    if os.name == "nt" or not dist.is_nccl_available():
        return "gloo"
    return "cpu:gloo,cuda:nccl"
