import gc
from typing import Dict, List, Optional, Union

import torch


def clear_gpu_memory(device_ids=None):
    """
    Thoroughly clears GPU memory across specified devices.

    Parameters
    ----------
    device_ids : list, optional
        List of GPU device IDs to clear. If None, clears all available devices.
    """
    try:
        # Run Python garbage collector
        for param in [
            p
            for p in gc.get_objects()
            if isinstance(p, torch.Tensor) and p.grad is not None
        ]:
            param.grad = None
        gc.collect()
        # If no specific devices specified, get all available devices
        if device_ids is None and torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
        elif not torch.cuda.is_available():
            return
        # Clear memory for each device
        for device_id in device_ids:
            try:
                # Set current device
                torch.cuda.set_device(device_id)
                # Clear the current device
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # current_memory = torch.cuda.memory_allocated(device_id)
                # max_memory = torch.cuda.max_memory_allocated(device_id)
                # print(
                #     f"Device {device_id} - Current memory: {current_memory/1e6:.2f}MB, Peak memory: {max_memory/1e6:.2f}MB"
                # )
            except RuntimeError as e:
                pass
    except Exception as e:
        raise


def get_gpu_memory_info() -> Optional[List[Dict[str, Union[int, str, float]]]]:
    """
    Returns detailed GPU memory information for all available devices.
    """
    if not torch.cuda.is_available():
        return None
    memory_info = []
    for i in range(torch.cuda.device_count()):
        current_memory = torch.cuda.memory_allocated(i)
        max_memory = torch.cuda.max_memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
        device_name = torch.cuda.get_device_name(i)
        # This is a dictionary with device ID, device name,
        #   current memory, peak memory, and cached memory
        info = {
            "device_id": i,
            "device_name": device_name,
            "current_memory_mb": current_memory / 1e6,
            "peak_memory_mb": max_memory / 1e6,
            "cached_memory_mb": cached_memory / 1e6,
        }
        memory_info.append(info)
    return memory_info
