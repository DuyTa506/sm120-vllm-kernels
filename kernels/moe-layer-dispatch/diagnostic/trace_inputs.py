"""Describe optional autotuner inputs without touching tensor values or devices."""
def describe(value):
    if value is None:
        return None
    if hasattr(value, 'shape'):
        return dict(kind='tensor', shape=list(value.shape), dtype=str(getattr(value,'dtype',None)))
    if isinstance(value, (str, int, float, bool)):
        return dict(kind=type(value).__name__, value=value)
    return dict(kind=type(value).__name__)
