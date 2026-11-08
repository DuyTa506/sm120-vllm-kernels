from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


class TraceGateTests(unittest.TestCase):
    def run_case(self,symbol,graph,profile):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
                db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (demangledName INTEGER,graphNodeId INTEGER)')
                db.execute('INSERT INTO StringIds VALUES (1,?)',(symbol,))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1,?)',(graph,))
            return subprocess.run([sys.executable,str(Path(__file__).with_name('check_trace.py')),str(path),profile,'1'],capture_output=True).returncode

    def test_eager_only_does_not_prove_captured_activation(self):
        self.assertNotEqual(self.run_case('skinny_fp8_kernel',None,'fp8'),0)
        self.assertEqual(self.run_case('skinny_fp8_kernel',7,'fp8'),0)

    def test_disabled_kernel_is_rejected(self):
        self.assertNotEqual(self.run_case('_skinny_bf16_gemm_kernel',7,'off'),0)
        self.assertEqual(self.run_case('cutlass_kernel',7,'off'),0)
