"""Read actual hybrid cache groups at startup, without GPU tensor reads."""
import dataclasses
import math

def encode_spec_field(value):
    """Only encode explicitly understood non-JSON cache-spec metadata."""
    from enum import Enum
    from fractions import Fraction
    import torch
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, Enum):
        return {"enum_type": type(value).__module__ + "." + type(value).__qualname__,
                "name": value.name, "value": value.value}
    if isinstance(value, Fraction):
        return {"fraction_numerator": value.numerator, "fraction_denominator": value.denominator}
    raise TypeError("unsupported cache spec field type: " + type(value).__module__ + "." + type(value).__qualname__)

def describe(runner):
    cfg=runner.kv_cache_config
    groups=[]
    for index,group in enumerate(cfg.kv_cache_groups):
        spec=group.kv_cache_spec
        page=spec.page_size_bytes
        maximum=spec.max_memory_usage_bytes(runner.vllm_config)
        groups.append(dict(index=index,layers=list(group.layer_names),spec_class=type(spec).__name__,spec_fields=dataclasses.asdict(spec),page_size_bytes=page,max_request_memory_bytes=maximum,max_request_resident_blocks=math.ceil(maximum/page)))
    return dict(num_blocks=cfg.num_blocks,max_model_len=runner.vllm_config.model_config.max_model_len,groups=groups)


def validate_preempt_geometry(data):
    assert data['num_blocks']==50 and data['max_model_len']==16384
    groups=data['groups']
    full=[g for g in groups if g['spec_class']=='FullAttentionSpec']
    mamba=[g for g in groups if g['spec_class']=='MambaSpec']
    assert len(full)==1 and len(mamba)==3 and len(groups)==4, 'unexpected hybrid group geometry'
    assert all(g['spec_fields']['block_size']==2176 for g in groups)
    assert all(g['spec_fields']['mamba_cache_mode']=='align' and g['spec_fields']['num_speculative_blocks']==2 and g['spec_fields']['num_prefill_checkpoint_blocks']==1 for g in mamba), 'unexpected Mamba resident state budget'
    assert full[0]['max_request_resident_blocks']==8
    assert all(g['max_request_resident_blocks']==5 for g in mamba)
    # This is a capacity hypothesis, not proof of observed scheduler eviction.
    available=data['num_blocks']-1
    initial=1+sum(g['max_request_resident_blocks'] for g in mamba)
    grown=2+sum(g['max_request_resident_blocks'] for g in mamba)
    cohort=available//initial
    assert cohort>=2 and cohort*grown>available
