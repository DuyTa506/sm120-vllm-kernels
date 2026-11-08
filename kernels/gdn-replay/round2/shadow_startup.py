"""Bound diagnostic suspension to the known worker startup lifecycle."""
from contextlib import contextmanager

@contextmanager
def suspend_startup_shadow(replay):
    assert not replay._SHADOW_STARTUP_SUSPENDED, 'nested startup suspension'
    assert replay._REGISTRY is None or not replay._REGISTRY._shadow, 'preexisting shadow records'
    replay._SHADOW_STARTUP_SUSPENDED = True
    try:
        yield
    finally:
        replay._SHADOW_STARTUP_SUSPENDED = False
        assert replay._REGISTRY is None or not replay._REGISTRY._shadow, 'startup retained shadow records'
