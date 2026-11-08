import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from composition_guards import replace_once, verify_runtime_sources


class CompositionGuards(unittest.TestCase):
    def test_exact_replacement_count(self):
        self.assertEqual(replace_once('abc','b','x'),'axc')
        for text in ('ac','abbc'):
            with self.assertRaises(ValueError):
                replace_once(text,'b','x')

    def test_wrong_runtime_source_rejected_even_with_correct_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pkg=root/'vllm';pkg.mkdir()
            (pkg/'envs.py').write_text('actual')
            manifest=root/'composition.json'
            manifest.write_text(json.dumps(dict(generated_sha256={'vllm/envs.py':hashlib.sha256(b'expected').hexdigest()})))
            sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'active source differs'):
                verify_runtime_sources(manifest,sha,pkg)
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                verify_runtime_sources(manifest,'wrong',pkg)
