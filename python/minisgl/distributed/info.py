from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.distributed as dist

@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    # TODO(ljc)
    actor: str
    local_rank: int
    local_size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        # TODO(ljc)
        assert 0 <= self.local_rank < self.local_size

    def is_primary(self) -> bool:
        return self.rank == 0
    
    # TODO(ljc)
    def is_primary_actor(self) -> bool:
        return self.local_rank == 0
    
    def is_draft(self) -> bool:
        return self.actor == "draft"


_TP_INFO: DistributedInfo | None = None
_TP_GROUP: torch.distributed.ProcessGroup | None = None

# TODO(ljc)
# def set_tp_info(rank: int, size: int) -> None:
#     global _TP_INFO
#     if _TP_INFO is not None:
#         raise RuntimeError("TP info has been set")
#     _TP_INFO = DistributedInfo(rank, size)
def set_tp_info(rank: int, size: int, actor: str, local_rank: int, local_size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size, actor, local_rank, local_size)

def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


def set_tp_group(tp_group: torch.distributed.ProcessGroup) -> None:
    global _TP_GROUP
    if _TP_GROUP is not None:
        raise RuntimeError("TP group has been set")
    _TP_GROUP = tp_group

def get_tp_group() -> torch.distributed.ProcessGroup | None:
    return _TP_GROUP


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info", "set_tp_group", "get_tp_group"]