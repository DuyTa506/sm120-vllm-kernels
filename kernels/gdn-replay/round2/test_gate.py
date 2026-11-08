import tempfile
from pathlib import Path
import unittest
from check_gate import evaluate, metric


class Gates(unittest.TestCase):
    def test_missing_coverage_and_preemption_fail(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            runtime=p/'serving/round2_runtime';runtime.mkdir(parents=True)
            (runtime/'runtime.json').write_text('{"composition_sha256":"fixture","fp32_layers":30,"composition_sources_verified":true}')
            (p/'metrics_before.txt').write_text('vllm:num_preemptions_total 0\n')
            (p/'metrics_after.txt').write_text('vllm:num_preemptions_total 0\n')
            (p/'coverage.log').write_text('GDN replay ROUND2 rows=30 bad=0 boundary_rows=0 boundary_bad=0 armed_bad=0 out_bad=0')
            self.assertFalse(evaluate(p,'shadow')['passed'])
            self.assertFalse(evaluate(p,'preempt')['passed'])
            (p/'coverage.log').write_text('GDN replay ROUND2 rows=30 bad=0 boundary_rows=2 boundary_bad=0 armed_bad=0 out_bad=0')
            self.assertTrue(evaluate(p,'shadow')['passed'])
            (runtime/'runtime.json').write_text('{"composition_sha256":"fixture","fp32_layers":29,"composition_sources_verified":true}')
            self.assertFalse(evaluate(p,'shadow')['passed'])
            (runtime/'runtime.json').write_text('{"composition_sha256":"fixture","fp32_layers":30,"composition_sources_verified":true}')
            (p/'metrics_after.txt').write_text('vllm:num_preemptions_total 1\n')
            self.assertTrue(evaluate(p,'preempt')['passed'])
            (p/'coverage.log').write_text('GDN replay ROUND2 rows=30 bad=1 boundary_rows=2 boundary_bad=0 armed_bad=0 out_bad=0')
            self.assertFalse(evaluate(p,'preempt')['passed'])


class CounterEvidence(unittest.TestCase):
    def test_missing_or_invalid_counter_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'metrics.txt'
            for text in ('', 'unrelated 1\n', 'vllm:num_preemptions_total NaN\n',
                         'vllm:num_preemptions_total -1\n'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    metric(p, 'vllm:num_preemptions_total')
            p.write_text('vllm:num_preemptions_total{engine="0"} 0\n')
            self.assertEqual(metric(p, 'vllm:num_preemptions_total'), 0)

    def test_counter_reset_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / 'coverage.log').write_text('')
            (p / 'metrics_before.txt').write_text('vllm:num_preemptions_total 2\n')
            (p / 'metrics_after.txt').write_text('vllm:num_preemptions_total 1\n')
            with self.assertRaisesRegex(ValueError, 'decreased'):
                evaluate(p, 'preempt')
