from types import SimpleNamespace
import unittest
from shadow_startup import suspend_startup_shadow

class StartupScope(unittest.TestCase):
    def make(self):
        return SimpleNamespace(_SHADOW_STARTUP_SUSPENDED=False, _REGISTRY=SimpleNamespace(_shadow=[]))
    def test_restores_after_exception(self):
        r=self.make()
        with self.assertRaisesRegex(RuntimeError,'startup failed'):
            with suspend_startup_shadow(r):
                self.assertTrue(r._SHADOW_STARTUP_SUSPENDED)
                raise RuntimeError('startup failed')
        self.assertFalse(r._SHADOW_STARTUP_SUSPENDED)
    def test_does_not_discard_preexisting(self):
        r=self.make(); r._REGISTRY._shadow.append('real')
        with self.assertRaisesRegex(AssertionError,'preexisting'):
            with suspend_startup_shadow(r): pass
        self.assertEqual(r._REGISTRY._shadow,['real'])
        self.assertFalse(r._SHADOW_STARTUP_SUSPENDED)
    def test_pending_rejected_and_restored(self):
        r=self.make()
        with self.assertRaisesRegex(AssertionError,'retained'):
            with suspend_startup_shadow(r): r._REGISTRY._shadow.append('unexpected')
        self.assertEqual(r._REGISTRY._shadow,['unexpected'])
        self.assertFalse(r._SHADOW_STARTUP_SUSPENDED)
    def test_scope_ends_before_runtime(self):
        r=self.make()
        with suspend_startup_shadow(r): self.assertTrue(r._SHADOW_STARTUP_SUSPENDED)
        self.assertFalse(r._SHADOW_STARTUP_SUSPENDED)
        r._REGISTRY._shadow.append('runtime')
        self.assertEqual(r._REGISTRY._shadow,['runtime'])
