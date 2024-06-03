import logging
import os
import loguru
import time
import psutil
import torch

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_num_gpus():
    return int(os.environ.get("NUM_GPUS", "1"))

logger = loguru.logger

def log_dist(message: str,
             ranks: list[int] = ["all"],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks or ranks == ["all"]:
        if level == logging.INFO:
            logger.opt(depth=1).info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.opt(depth=1).error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.opt(depth=1).debug(f'[Rank {my_rank}] {message}')
        if level == logging.WARNING:
            logger.opt(depth=1).warning(f'[Rank {my_rank}] {message}')

def log_memory_usage(s="", level=logging.INFO):
    process = psutil.Process(os.getpid())
    Bytes_to_GB = 1024.0 * 1024.0 * 1024.0
    allocated = torch.cuda.memory_allocated() / Bytes_to_GB
    reserved = torch.cuda.memory_reserved() / Bytes_to_GB
    peak_allocated = torch.cuda.max_memory_allocated() / Bytes_to_GB
    peak_reserved = torch.cuda.max_memory_reserved() / Bytes_to_GB
    free, total = torch.cuda.mem_get_info()
    free = free / Bytes_to_GB
    total = total / Bytes_to_GB
    other_cuda_used = total - (free + reserved)
    mem_info = process.memory_info()
    vms: float = mem_info.vms / Bytes_to_GB
    rss: float = mem_info.rss / Bytes_to_GB
    shared: float = mem_info.shared / Bytes_to_GB
    free_m = int(os.popen('free -t -m').readlines()[1].split()[3]) / 1024.0
    log_dist(f"{s} allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak allocated={peak_allocated:.2f} GB, " + 
             f"peak reserved={peak_reserved:.2f} GB, cuda free={free:.2f} GB, cuda total={total:.2f} GB, other cuda used={other_cuda_used:.2f} GB, " + 
             f"ram free={free_m} GB, rss={rss:.2f} GB, vms={vms:.2f} GB, shared={shared:.2f} GB", level=level)