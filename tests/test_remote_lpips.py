"""Validate LPIPS argument gates and launcher plumbing without CUDA or weights."""

import argparse
import ast
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REMOTE = Path(__file__).parents[1] / "scripts/remote"


def load_parser():
    # The remote runner imports CUDA training code at module import time. Execute
    # its actual parser function alone so these CLI checks stay dependency-free.
    source = ast.parse((REMOTE / "csa_remote_train.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    namespace = {"argparse": argparse, "math": math}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(REMOTE / "csa_remote_train.py"),
            "exec",
        ),
        namespace,
    )
    return namespace["parse_args"]


class TestRemoteLpipsArguments(unittest.TestCase):
    def setUp(self):
        self.parse = load_parser()
        self.base = ["train", "--runtime", "/runtime", "--stage", "stage1"]

    def test_defaults_preserve_l1(self):
        args = self.parse(self.base)
        self.assertEqual(args.reconstruction_weight, 1.0)
        self.assertEqual(args.lpips_weight, 0.0)
        self.assertEqual(args.lpips_net, "vgg")
        self.assertEqual(args.lpips_batch_size, 8)
        self.assertFalse(args.allow_loss_change)

    def test_explicit_loss_transition(self):
        args = self.parse(
            self.base
            + [
                "--resume",
                "/runtime/run/latest.pt",
                "--reconstruction-weight",
                "0.1",
                "--lpips-weight",
                "1",
                "--allow-loss-change",
            ]
        )
        self.assertEqual(args.reconstruction_weight, 0.1)
        self.assertEqual(args.lpips_weight, 1.0)
        self.assertTrue(args.allow_loss_change)

    def test_reject_invalid_or_unsupported_lpips_arguments(self):
        cases = [
            (["--reconstruction-weight", "-0.1"], "finite and nonnegative"),
            (["--reconstruction-weight", "nan"], "finite and nonnegative"),
            (["--reconstruction-weight", "inf"], "finite and nonnegative"),
            (["--lpips-weight", "-0.1"], "finite and nonnegative"),
            (["--lpips-weight", "nan"], "finite and nonnegative"),
            (["--lpips-weight", "inf"], "finite and nonnegative"),
            (["--lpips-batch-size", "0"], "batch size must be positive"),
            (["--lpips-net", "random"], "invalid choice"),
            (["--allow-loss-change"], "requires --stage stage1 and --resume"),
            (
                ["--stage", "joint", "--resume", "/ck.pt", "--allow-loss-change"],
                "requires --stage stage1 and --resume",
            ),
            (["--stage", "joint", "--lpips-weight", "0.1"], "only with --stage stage1"),
            (["--diagnostic"], "does not allow random perceptual weights"),
        ]
        for extra, error in cases:
            with self.subTest(extra=extra):
                output = io.StringIO()
                with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as caught:
                    self.parse(self.base + extra)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(error, output.getvalue())


class TestRemoteLpipsLauncher(unittest.TestCase):
    def run_launcher(self, extra_env, resume=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "run/runtime"
            runtime.mkdir(parents=True)
            commands = root / "commands.jsonl"
            fake_python = root / "fake-python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['CSA_TEST_COMMANDS'], 'a') as output:\n"
                "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_python.chmod(0o755)
            fake_flock = root / "flock"
            fake_flock.write_text("#!/bin/sh\nexit 0\n")
            fake_flock.chmod(0o755)
            environment = dict(os.environ)
            for name in list(environment):
                if name.startswith("CSA_"):
                    del environment[name]
            environment.update(
                PATH=str(root) + os.pathsep + environment["PATH"],
                CSA_GPU_UUID="GPU-test",
                CSA_TEST_COMMANDS=str(commands),
                **extra_env,
            )
            command = [
                "bash",
                str(REMOTE / "launch_stage1.sh"),
                str(runtime),
                str(fake_python),
                "256",
                "12",
            ]
            if resume:
                command.append(str(runtime / "run/latest.pt"))
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            calls = (
                [json.loads(line) for line in commands.read_text().splitlines()]
                if commands.exists()
                else []
            )
            return result, calls

    def test_probe_and_training_share_lpips_settings(self):
        result, calls = self.run_launcher(
            {
                "CSA_RECONSTRUCTION_WEIGHT": "0.1",
                "CSA_LPIPS_WEIGHT": "1",
                "CSA_LPIPS_NET": "vgg",
                "CSA_LPIPS_BATCH_SIZE": "8",
                "CSA_ALLOW_LOSS_CHANGE": "1",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        probe = next(call for call in calls if "probe" in call)
        train = next(call for call in calls if "train" in call)
        for call in [probe, train]:
            self.assertEqual(call[call.index("--reconstruction-weight") + 1], "0.1")
            self.assertEqual(call[call.index("--lpips-weight") + 1], "1")
            self.assertEqual(call[call.index("--lpips-net") + 1], "vgg")
            self.assertEqual(call[call.index("--lpips-batch-size") + 1], "8")
            self.assertEqual(call[call.index("--batch-size") + 1], "256")
        self.assertNotIn("--allow-loss-change", probe)
        self.assertNotIn("--resume", probe)
        self.assertIn("--allow-loss-change", train)
        self.assertIn("--resume", train)

    def test_loss_transition_flag_defaults_off(self):
        result, calls = self.run_launcher({})
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in calls:
            self.assertNotIn("--allow-loss-change", call)
        train = next(call for call in calls if "train" in call)
        self.assertEqual(train[train.index("--reconstruction-weight") + 1], "1")
        self.assertEqual(train[train.index("--lpips-weight") + 1], "0")

    def test_loss_transition_needs_checkpoint_before_probe(self):
        result, calls = self.run_launcher({"CSA_ALLOW_LOSS_CHANGE": "1"}, resume=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a resume checkpoint", result.stderr)
        self.assertEqual(calls, [])

    def test_invalid_loss_transition_flag_rejected_before_probe(self):
        result, calls = self.run_launcher({"CSA_ALLOW_LOSS_CHANGE": "yes"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 0 or 1", result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
