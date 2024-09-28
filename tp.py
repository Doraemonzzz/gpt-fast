# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
if os.uname().sysname != "Darwin":
    from torch.distributed import _functional_collectives as funcol
else:
    # Distributed is not supported on MacOS
    funcol = None

from model import Attention, FeedForward, Transformer
from quantize import WeightOnlyInt4Linear

from xopes.ops import parallel_gumbel_multinomial_triton, gumbel_multinomial_reduce_triton

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_local():
    return _get_rank() == 0

def local_break():
    if is_local():
        breakpoint()
    dist.barrier()

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def _apply_tp_linear(linear: nn.Linear, style: str, weight_splits: List[int] = []) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q,k,v), dim=dim)

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        if isinstance(linear, WeightOnlyInt4Linear):
            sharded_weight = shard_qkv(linear.weight, shard_dim, [i//8 for i in weight_splits])
            linear.scales_and_zeros = shard_qkv(linear.scales_and_zeros, 1 - shard_dim, weight_splits)
        else:
            sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if isinstance(linear, WeightOnlyInt4Linear):
            linear.scales_and_zeros = shard(linear.scales_and_zeros, 1 - shard_dim)
            if style == "rowwise":
                assert linear.scales_and_zeros.shape[0] * 32 == sharded_weight.shape[1] * sharded_weight.shape[2] * sharded_weight.shape[3]
                assert linear.scales_and_zeros.shape[1] == sharded_weight.shape[0] * 8
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    # local_break()
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_ffn(mlp: nn.Linear) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")

    _apply_tp_linear(mlp.w1, "colwise")
    _apply_tp_linear(mlp.w3, "colwise")
    _apply_tp_linear(mlp.w2, "rowwise")

    world_size = _get_world_size()
    mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
        output, "sum", list(range(world_size))))


def _apply_tp_attn(attn: Attention) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")

    kv_size = attn.n_local_heads * attn.head_dim
    _apply_tp_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size])
    _apply_tp_linear(attn.wo, "rowwise")

    # overwrite
    world_size = _get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size

    attn.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
        output[0], "sum", list(range(world_size))))

def _apply_tp_lm_head(out: nn.Linear, parallel_lm_head=False) -> None:
    rank = _get_rank()
    world_size = _get_world_size()
    tgt = world_size - 1
    
    class ColumnParallelLinear(torch.nn.Module):
        def __init__(
            self,
            weight,
        ):
            super().__init__()
            
            def shard(x, dim):
                assert x.size(dim=dim) % world_size == 0
                return torch.tensor_split(x, world_size, dim=dim)[rank]
            
            if parallel_lm_head:
                sharded_weight = shard(weight, 0)
                self.weight = nn.Parameter(sharded_weight, requires_grad=False)
            else:
                self.weight = nn.Parameter(weight, requires_grad=False)
                
            self.in_features = self.weight.shape[1]
            self.out_features = self.weight.shape[0]
            self.parallel_lm_head = parallel_lm_head

        def forward(self, input, use_flash=False, is_prefill=False):
            if use_flash:
                if is_prefill:
                    input = input[:, -1:]
                    
                if self.parallel_lm_head:
                    idx, lse = parallel_gumbel_multinomial_triton(input, self.weight.transpose(0, 1), output_lse=True)
                    
                    world_size = _get_world_size()
                    rank = _get_rank()

                    idx_list = None
                    if rank == tgt:
                        idx_list = [torch.empty_like(idx) for _ in range(world_size)]
                        
                    lse_list = None
                    if rank == tgt:
                        lse_list = [torch.empty_like(lse) for _ in range(world_size)]
                        
                    dist.gather(idx, idx_list, dst=tgt)
                    dist.gather(lse, lse_list, dst=tgt)
                    
                    if rank == tgt:
                        # print(idx.shape, lse.shape)
                        # b k -> b k m
                        idx = torch.stack(idx_list, dim=-1)
                        # b 1 -> b m
                        lse = torch.cat(lse_list, dim=-1)
                        
                        # print(idx.shape, lse.shape)
                        
                        idx_next = gumbel_multinomial_reduce_triton(idx, lse)
                        
                        # print("aaa", idx.shape, idx_next.shape)
                    else:
                        idx_next = idx
                else:
                    if is_prefill:
                        input = input[:, -1:]
                    idx_next, lse = parallel_gumbel_multinomial_triton(input, self.weight.transpose(0, 1))
                    
                return idx_next
            else:
                output = F.linear(input, self.weight)
                
                if self.parallel_lm_head: # if parallel over vocab, do gather
                    world_size = _get_world_size()
                    rank = _get_rank()
                    
                    output_list = None
                    if rank == tgt:
                        output_list = [torch.empty_like(output) for _ in range(world_size)]
                    dist.gather(output, output_list, dst=tgt)
                    torch.cuda.synchronize()

                    if rank == tgt:
                        output = torch.cat(output_list, dim=-1)
                    
                    # output_list = [torch.empty_like(output) for _ in range(world_size)]
                    # dist.all_gather(output_list, output)
                    # output = torch.cat(output_list, dim=-1)

                return output

        def extra_repr(self) -> str:
            return f'in_features={self.in_features}, out_features={self.out_features}'
            
    out_col_parallel = ColumnParallelLinear(out.weight)
    
    return out_col_parallel

def _apply_tp_Transformer(Transformer: Transformer, parallel_lm_head=False) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = _get_world_size()
    Transformer.config.n_head = Transformer.config.n_head // world_size
    Transformer.config.dim = Transformer.config.dim // world_size
    Transformer.config.n_local_heads = Transformer.config.n_local_heads // world_size
    Transformer.output = _apply_tp_lm_head(Transformer.output, parallel_lm_head)
    Transformer.use_tp = True

def apply_tp(model: Transformer, parallel_lm_head=False) -> None:
    print(model)
    _apply_tp_Transformer(model, parallel_lm_head)
    for block in model.layers:
        # Apply to MLP
        _apply_tp_ffn(block.feed_forward)
        _apply_tp_attn(block.attention)
        
    print(model)
