"""Opt-in immutable per-owner/per-layer B12x workspace token bands.
Only source preparation and workspace selection live here; no native kernel edits.
"""
import ast
import hashlib
from pathlib import Path

FLAG='VLLM_SM120_WORKSPACE_BANDS'
BANDS=((1024,1535,1535,32),(1536,3071,3071,64))

def capacities(max_tokens):
    if type(max_tokens) is not int or max_tokens<3072:
        raise ValueError('Workspace bands require current parent max_num_tokens >=3072')
    return ((1535,32),(3071,64),(max_tokens,128))

def choose_band(tokens,max_tokens):
    choices=capacities(max_tokens)
    if type(tokens) is not int or not 1024<=tokens<=max_tokens:
        raise ValueError('Native workspace bands only accept1024..max_num_tokens; retain outer CUTLASS gate')
    return choices[0] if tokens<=1535 else choices[1] if tokens<=3071 else choices[2]

def ensure_wrapper(expert,tokens,*,wrapper_cls=None,manager=None):
    """Prepare all bands before warmup lock, then return an immutable selected band.

    The WorkspaceManager supplies owner/ubatch/lane isolation. Never retain an
    owner-specific wrapper on the expert instance and never mutate a workspace.
    Within one owner calls remain serialized, matching parent scratch semantics.
    """
    selected=choose_band(tokens,expert.max_num_tokens)
    if (expert.global_num_experts,expert.num_local_experts,expert.topk,expert.hidden_dim,expert.intermediate_size_per_partition)!=(256,256,8,2048,512):
        raise ValueError('Unvalidated shape/expert-parallel contract for workspace bands')
    if getattr(expert,'_hybrid_min_tokens',None)!=1024:
        raise ValueError('Workspace-band adapter requires unchanged1024-token outer native threshold')
    if wrapper_cls is None:
        from flashinfer.fused_moe import B12xMoEWrapper
        wrapper_cls=B12xMoEWrapper
    if manager is None:
        from vllm.v1.worker.workspace import current_workspace_manager
        manager=current_workspace_manager()
    if not hasattr(manager,'get_persistent_resource'):
        raise RuntimeError('WorkspaceManager persistent owner resources required')
    chosen=None
    for capacity,tile in capacities(expert.max_num_tokens):
        config=dict(num_experts=expert.global_num_experts,top_k=expert.topk,hidden_size=expert.hidden_dim,
                    intermediate_size=expert.intermediate_size_per_partition,use_cuda_graph=True,max_num_tokens=capacity,
                    num_local_experts=expert.num_local_experts,activation=expert._activation_str,
                    output_dtype=expert.out_dtype,device=expert.w1_scale.device)
        def create_layer_wrapper(config=config,capacity=capacity):
            seed=manager.get_persistent_resource((wrapper_cls,'buffers-token-bands-v1',capacity,tuple(config.items())),lambda:wrapper_cls(**config))
            return wrapper_cls(**config,shared_static_workspace=seed._static_workspace,
                shared_dynamic_workspace=seed._dynamic_workspace,shared_output=seed._moe_output)
        wrapper=manager.get_persistent_resource((type(expert),'wrapper-token-bands-v1',expert,capacity,tuple(config.items())),create_layer_wrapper)
        if wrapper._dynamic_workspace.tile_m!=tile:
            raise RuntimeError(f'Native allocated tile changed: capacity{capacity} expected{tile} got{wrapper._dynamic_workspace.tile_m}')
        if (capacity,tile)==selected:chosen=wrapper
    assert chosen is not None
    return chosen

def compose_experts_source(source,expected_sha256):
    """Produce isolated source bytes; never change the shared original file."""
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha256:
        raise ValueError('Parent scalar source hash mismatch')
    signature='    def _ensure_wrapper(self) -> Any:'
    call='        wrapper = self._ensure_wrapper()'
    if source.count(signature)!=1 or source.count(call)!=1 or FLAG in source:
        raise ValueError('Unexpected parent integration boundary')
    replacement='''    def _ensure_wrapper(self, tokens: int | None = None) -> Any:
        from vllm import envs
        if envs.VLLM_SM120_WORKSPACE_BANDS:
            if not _B12X_SHARED_WORKSPACE:
                raise RuntimeError("Workspace bands require parent shared workspace")
            from workspace_adapter import ensure_wrapper
            return ensure_wrapper(self, tokens)
'''
    candidate=source.replace(signature,replacement.rstrip('\n')).replace(call,'        wrapper = self._ensure_wrapper(hidden_states.shape[0])')
    ast.parse(candidate)
    return candidate

def compose_envs_source(source):
    if FLAG in source:raise ValueError('Workspace bands already registered')
    candidate=source+f'\n# Immutable workspace geometry participates in compile identity.\nenvironment_variables[{FLAG!r}] = lambda: os.getenv({FLAG!r}, "0") == "1"\n'
    ast.parse(candidate)
    return candidate
