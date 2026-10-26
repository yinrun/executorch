import torch


def snapshot_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "musa":
        return tensor.detach().to(device="cpu", copy=True).contiguous()
    with torch.musa.device(tensor.device):
        torch.musa.synchronize(tensor.device)
        source = tensor.detach().contiguous()
        staging = torch.empty(
            source.shape, dtype=source.dtype, device="cpu", pin_memory=True
        )
        staging.copy_(source, non_blocking=True)
        torch.musa.synchronize(tensor.device)
        return staging.clone()
