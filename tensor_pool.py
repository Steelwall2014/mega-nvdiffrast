import os
import torch
from collections import defaultdict
import time

def default_factory():
    return []
class tensor_pool:
    # tensor pool for float32 cpu tensors
    def __init__(self):
        self.allocated = set()
        self.reserved = defaultdict(default_factory)

    def allocate(self, shape, pin_memory=False):
        if len(self.reserved[shape]) > 0:
            tensor = self.reserved[shape].pop()
        else:
            tensor = torch.empty(shape, dtype=torch.float32, pin_memory=pin_memory, device="cpu")

        self.allocated.add(tensor)
        return tensor
        
    def release(self, tensor: torch.Tensor):
        # assert tensor in self.allocated
        if tensor not in self.allocated:
            return
        self.allocated.remove(tensor)
        self.reserved[tensor.shape].append(tensor)

    def info(self):
        allocated = sum([t.numel() * t.element_size() for t in self.allocated]) / 1024 / 1024
        num_allocated = len(self.allocated)
        reserved = 0
        for shape, tensors in self.reserved.items():
            for t in tensors:
                reserved += t.numel() * t.element_size()
        reserved = reserved / 1024 / 1024
        num_reserved = sum([len(tensors) for tensors in self.reserved.values()])
        return {"allocated (MB)": allocated, "reserved (MB)": reserved, "num_allocated": num_allocated, "num_reserved": num_reserved}

_pool = tensor_pool()
def allocate(shape, pin_memory=False) -> torch.Tensor:
    return _pool.allocate(shape, pin_memory)
def release(tensor: torch.Tensor):
    return _pool.release(tensor)
def to_cpu(tensor: torch.Tensor, pin_memory=False) -> torch.Tensor:
    if tensor is None:
        return None
    shape = tensor.shape
    cpu_tensor = allocate(shape, pin_memory)
    cpu_tensor.copy_(tensor, non_blocking=True)
    return cpu_tensor