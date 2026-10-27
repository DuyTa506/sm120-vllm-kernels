"""CPU subprocess checks with fake Docker; never starts a container."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probe = self.root / "kernels/decode-route-capture"
        self.probe.mkdir(parents=True)
        shutil.copy(Path(__file__).with_name("run.sh"), self.probe / "run.sh")
        (self.probe / "run_capture.py").write_text("# unused by fake Docker\n")
        for relative in ("v1/attention/backends/flashinfer.py",
                         "v1/attention/backends/sm120_fp8_fmha_glue.py",
                         "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                         "model_executor/layers/mamba/ops/fused_conv_post_conv.py"):
            path = self.root / "build/vllm" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "docker"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n'
                        'if [ "$1" = run ]; then\n'
                        '  while [ "$#" -gt 0 ]; do\n'
                        '    if [ "$1" = --cidfile ] && [ -n "${FAKE_CID:-}" ]; then\n'
                        '      printf "%s\\n" "$FAKE_CID" > "$2"\n'
                        '    fi\n'
                        '    shift\n'
                        '  done\n'
                        '  exit "${FAKE_DOCKER_EXIT:-0}"\nfi\n')
        fake.chmod(0o755)
        smi = self.bin / "nvidia-smi"
        smi.write_text('#!/bin/sh\ncase "$*" in\n'
                       '  *--query-compute-apps=pid*) printf "%s" "${FAKE_GPU_PIDS:-}" ;;\n'
                       '  *) printf "fake CPU-test telemetry\\n" ;;\nesac\n')
        smi.chmod(0o755)
        self.lock = self.root / "gpu.lock"
        self.log = self.root / "docker.log"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        GPU_LOCK=str(self.lock), FAKE_DOCKER_LOG=str(self.log))
        self.env.pop("GPU_LOCK_HELD", None)

    def run_launcher(self, borrowed):
        return subprocess.run(["bash", str(self.probe / "run.sh"), "test", "--control",
                               "--concurrency", "1", "--output-tokens", "32"],
                              env=dict(self.env, GPU_LOCK_HELD=str(borrowed)),
                              text=True, capture_output=True, timeout=5)

    def test_standalone_and_only_the_requested_driver_runs(self):
        result = self.run_launcher(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("image inspect", calls[0])
        self.assertIn("/work/kernels/decode-route-capture/run_capture.py", calls[1])
        self.assertIn("--control --concurrency 1 --output-tokens 32", calls[1])
        self.assertNotIn("agentic_bench", calls[1])
        out = self.probe / "results/test"
        self.assertEqual((out / "exit_code.txt").read_text().strip(), "0")
        self.assertTrue((out / "gpu_before.csv").exists())
        self.assertTrue((out / "gpu_after.csv").exists())

    def test_busy_gpu_is_refused_without_starting_docker(self):
        self.env["FAKE_GPU_PIDS"] = "12345\n"
        result = self.run_launcher(0)
        self.assertEqual(result.returncode, 3)
        self.assertFalse(self.log.exists())
        out = self.probe / "results/test"
        self.assertEqual((out / "exit_code.txt").read_text().strip(), "3")

    def test_failed_run_retains_original_exit_code_and_after_telemetry(self):
        self.env["FAKE_DOCKER_EXIT"] = "17"
        result = self.run_launcher(0)
        self.assertEqual(result.returncode, 17)
        out = self.probe / "results/test"
        self.assertEqual((out / "exit_code.txt").read_text().strip(), "17")
        self.assertTrue((out / "gpu_after.csv").exists())

    def test_cleanup_only_removes_the_container_created_by_this_invocation(self):
        self.env.update(FAKE_CID="abcdef123456", FAKE_DOCKER_EXIT="17")
        result = self.run_launcher(0)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.log.read_text().splitlines()[-1], "rm -f abcdef123456")

    def test_borrowed_parent_lock_does_not_deadlock_or_release_parent(self):
        with self.lock.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            result = self.run_launcher(1)
            self.assertEqual(result.returncode, 0, result.stderr)
            with self.lock.open("w") as second:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_borrowed_mode_rejects_an_unlocked_file(self):
        result = self.run_launcher(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_invalid_lock_mode_rejected_before_docker(self):
        result = self.run_launcher("yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
