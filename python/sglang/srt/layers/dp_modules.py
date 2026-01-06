from __future__ import annotations

import torch
import logging
from typing import TYPE_CHECKING, Optional, List
from enum import IntEnum, auto

from sglang.srt.distributed import (
    GroupCoordinator,
    get_tp_group,
)

from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_size,
    get_dp_local_info,
    memcpy_triton,
)

class DpModuleGroup:
    def __init__(self, tp_group, tp_rank, tp_size, dp_rank, dp_size):
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size

    def __str__(self):
        return f"dp_size={self.dp_size} tp_size={self.tp_size} current: DP{self.dp_rank} TP{self.tp_rank}"


class DpModuleId(IntEnum):
    # ATTENTION = 0
    MOE_DENSE = 1
    MOE_SHARED_EXPERT= 2
    EMBEDDING = 3
    LM_HEAD = 4

    @classmethod
    def max_value(cls):
        return max(member.value for member in cls)
    
    @classmethod
    def to_list(cls):
        return [member.value for member in cls]


class FFNInputMode(IntEnum):
    ORIGIN = auto()
    SCATTERED = auto()
    GATHERED = auto()
    
    @classmethod
    def gen_mode(self, module_tp_size):
        if module_tp_size == get_attention_tp_size():
            return FFNInputMode.ORIGIN
        elif module_tp_size > get_attention_tp_size():
            return FFNInputMode.GATHERED
        else:
            return FFNInputMode.SCATTERED



_DP_MODULE_GROUPS_ = [None for id in range(DpModuleId.max_value() + 1)]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def initialize_dp_for_module(
    world_tp_rank: int,
    world_tp_size: int,
    attention_tp_size: int,
    module_tp_size: int,
    group_name: str,
):
    if module_tp_size <= 0 or module_tp_size > world_tp_size:
        logger.error(f'## invalid {module_tp_size=} for {group_name=}, world_size={world_tp_size}')
        return DpModuleGroup(None, None, None, None, None)

    if module_tp_size == world_tp_size:
        return DpModuleGroup(get_tp_group(), world_tp_rank, world_tp_size, 0, 1)
    
    if module_tp_size == attention_tp_size:
        return DpModuleGroup(get_attention_tp_group(), get_attention_tp_rank(), attention_tp_size, get_attention_dp_rank(), get_attention_dp_size())
    
    module_dp_size = world_tp_size // module_tp_size
    module_dp_rank = world_tp_rank // module_tp_size
    module_tp_rank = world_tp_rank % module_tp_size

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP
    tp_group = get_tp_group()
    module_tp_group = GroupCoordinator(
        [
            list(range(head, head + module_tp_size))
            for head in range(0, world_tp_size, module_tp_size)
        ],
        tp_group.local_rank,
        torch.distributed.get_backend(tp_group.device_group),
        SYNC_TOKEN_IDS_ACROSS_TP,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name=group_name,
    )
    return DpModuleGroup(module_tp_group, module_tp_rank, module_tp_size, module_dp_rank, module_dp_size)


def initialize_dp_modules(
    tp_rank: int,
    tp_size: int,
    moe_dense_tp_size: Optional[int] = None,
    moe_shared_expert_tp_size: Optional[int] = None,
    embedding_tp_size: Optional[int] = None,
    lmhead_tp_size: Optional[int] = None,
):
    global _DP_MODULE_GROUPS_

    module_tp_size_list = list(set(
        [x for x in [moe_dense_tp_size, moe_shared_expert_tp_size, embedding_tp_size, lmhead_tp_size] if x is not None]))
    attention_tp_size = get_attention_tp_size()
    dp_groups = {}
    for module_tp_size in module_tp_size_list:
        dp_groups[module_tp_size] = initialize_dp_for_module(
            tp_rank, tp_size, attention_tp_size, module_tp_size, f'special_{module_tp_size}_tp_group')
        
    if moe_dense_tp_size is not None:
        _DP_MODULE_GROUPS_[DpModuleId.MOE_DENSE] = dp_groups[moe_dense_tp_size]
        # logger.error(f'moe_dense_tp_group: {str(_DP_MODULE_GROUPS_[DpModuleId.MOE_DENSE])}')
    if moe_shared_expert_tp_size is not None:
        _DP_MODULE_GROUPS_[DpModuleId.MOE_SHARED_EXPERT] = dp_groups[moe_shared_expert_tp_size]
        # logger.error(f'moe_shared_expert_tp_group: {str(_DP_MODULE_GROUPS_[DpModuleId.MOE_SHARED_EXPERT])}')
    if embedding_tp_size is not None:
        _DP_MODULE_GROUPS_[DpModuleId.EMBEDDING] = dp_groups[embedding_tp_size]
        # logger.error(f'embedding_tp_group: {str(_DP_MODULE_GROUPS_[DpModuleId.EMBEDDING])}')
    if lmhead_tp_size is not None:
        _DP_MODULE_GROUPS_[DpModuleId.LM_HEAD] = dp_groups[lmhead_tp_size]
        # logger.error(f'lm_head_tp_group: {str(_DP_MODULE_GROUPS_[DpModuleId.LM_HEAD])}')


def get_module_tp_group(module_id: DpModuleId):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_group


def get_module_tp_rank(module_id: DpModuleId):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_rank


def get_module_tp_size(module_id: DpModuleId):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_size


def get_module_dp_rank(module_id: DpModuleId):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].dp_rank


def get_module_dp_size(module_id: DpModuleId):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].dp_size


def module_tp_all_reduce(module_id: DpModuleId, input_: torch.Tensor) -> torch.Tensor:
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_group.all_reduce(input_)


def module_tp_all_gather(
    module_id: DpModuleId,
    input_: torch.Tensor,
    dim: int = -1,
    tensor_list: List[torch.Tensor] = None,
) -> torch.Tensor:
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_group.all_gather(input_, dim, tensor_list)


def module_tp_reduce_scatter(
    module_id: DpModuleId,
    output: torch.Tensor,
    input_list: List[torch.Tensor],
):
    assert _DP_MODULE_GROUPS_[module_id] is not None, f"dp {module_id} not initialized!"
    return _DP_MODULE_GROUPS_[module_id].tp_group.reduce_scatter(output, input_list)


def get_module_local_tokens(module_id: DpModuleId, forward_batch: ForwardBatch):
    module_dp_rank = get_module_dp_rank(module_id)
    module_tp_size = get_module_tp_size(module_id)
    step = module_tp_size // get_attention_tp_size()
    return forward_batch.global_num_tokens_cpu[module_dp_rank * step : (module_dp_rank + 1) * step]


def get_module_attn_info(module_id: DpModuleId, forward_batch: ForwardBatch):
    if forward_batch.dp_module_start_pos is None:
        cumtokens_gpu = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)

        forward_batch.dp_module_start_pos = torch.zeros(
            (DpModuleId.max_value() + 1, ),
            device=cumtokens_gpu.device,
            dtype=cumtokens_gpu.dtype,
        )
        forward_batch.dp_module_num_tokens = torch.zeros_like(forward_batch.dp_module_start_pos)

        attn_dp_size = get_attention_dp_size()
        for module in DpModuleId.to_list():
            if _DP_MODULE_GROUPS_[module] is None:
                continue
            module_dp_rank = get_module_dp_rank(module)
            module_dp_size =  get_module_dp_size(module)
            
            if module_dp_size == attn_dp_size:
                continue

            if module_dp_size < attn_dp_size:
                assert (attn_dp_size % module_dp_size == 0), f'{attn_dp_size=} must be mutiple of {module_dp_size=} '
                step = attn_dp_size // module_dp_size
                if module_dp_rank == 0:
                    forward_batch.dp_module_num_tokens[module] = cumtokens_gpu[(module_dp_rank + 1) * step - 1]
                else:
                    forward_batch.dp_module_start_pos[module] = cumtokens_gpu[module_dp_rank * step - 1]
                    forward_batch.dp_module_num_tokens[module] = cumtokens_gpu[(module_dp_rank + 1) * step - 1] - forward_batch.dp_module_start_pos[module]
            else:
                # assert (module_dp_size % attn_dp_size == 0), f'{module_dp_size=} must be mutiple of {attn_dp_size=} '
                # step = module_dp_size // attn_dp_size
                # token_list = forward_batch.global_num_tokens_gpu[attn_dp_rank].tensor_split(step)
                # module_dp_rank % 
                continue
            
    return forward_batch.dp_module_start_pos[module_id], forward_batch.dp_module_num_tokens[module_id]

# def get_attn_module_info(module_id: DpModuleId, scattered_tokens: torch.Tensor):
#     cumtokens = torch.cumsum(scattered_tokens, dim=0)
#     module_dp_rank = get_module_dp_rank(module_id)
#     step = get_module_dp_size(module_id) // get_attention_dp_size()
#     if module_dp_rank == 0:
#         local_start_pos = torch.zeros_like(cumtokens[0])
#     else:
#         local_start_pos = cumtokens[module_dp_rank % step - 1]
#     local_num_tokens = scattered_tokens[module_dp_rank % step]
#     return local_start_pos, local_num_tokens


def _module_gather_attn(
    module_id: DpModuleId,
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    is_nextn_dp: bool = False,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
        assert (
            global_tokens.untyped_storage().data_ptr()
            != local_tokens.untyped_storage().data_ptr()
        ), "aliasing between global_tokens and local_tokens not allowed"
        if is_nextn_dp or forward_batch.forward_mode.is_draft_extend():
            shape_tensor = local_num_tokens.new_full((), local_tokens.shape[0])
            local_num_tokens = torch.minimum(local_num_tokens, shape_tensor)

        start_pos = local_start_pos
        if get_module_dp_rank(module_id) != 0:
            module_offset_to_attn, _ = get_module_attn_info(module_id, forward_batch)
            start_pos = local_start_pos - module_offset_to_attn

        # logger.error(f'## _dp_module_gather {local_tokens.shape=} {global_tokens.shape=} {local_start_pos=} {local_num_tokens=}')
        memcpy_triton(
            global_tokens, local_tokens, 0, start_pos, local_num_tokens, False
        )

    # only all reduce in own tp group
    # Input IDs are in int 32. We should use inplace_all_reduce for local case becaues of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_module_tp_group(module_id).world_size <= NUM_GPUS_PER_NODE
    ):
        torch.ops.sglang.inplace_all_reduce(
            global_tokens, group_name=get_module_tp_group(module_id).unique_name
        )
    else:
        global_tokens[:] = module_tp_all_reduce(module_id, global_tokens)


def module_gather_attn_partial(
    module_id: DpModuleId,
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_nextn_dp: bool = False,
):
    _module_gather_attn(module_id, global_tokens, local_tokens, forward_batch, is_partial=True, is_nextn_dp=is_nextn_dp)


def module_gather_attn_replicate(
    module_id: DpModuleId,
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _module_gather_attn(module_id, global_tokens, local_tokens, forward_batch, is_partial=False)


def module_scatter_attn(
    module_id: DpModuleId,
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
    is_nextn_dp: bool = False,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage().data_ptr()
            != global_tokens.untyped_storage().data_ptr()
        ), "aliasing between local_tokens and global_tokens not allowed"

        if is_nextn_dp or forward_batch.forward_mode.is_draft_extend():
            shape_tensor = local_num_tokens.new_full((), local_tokens.shape[0])
            local_num_tokens = torch.minimum(local_num_tokens, shape_tensor)

        start_pos = local_start_pos
        if get_module_dp_rank(module_id) != 0:
            module_offset_to_attn, _ = get_module_attn_info(module_id, forward_batch)
            start_pos = local_start_pos - module_offset_to_attn
        memcpy_triton(
            local_tokens, global_tokens, 0, start_pos, local_num_tokens, True
        )