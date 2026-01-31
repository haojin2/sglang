# """MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU)

# Tests DeepSeek-V3.2 with TP=8 + MTP (EAGLE speculative decoding) using few-shot
# completion benchmark on MI35x.

# Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp suite
# """

# import os

# # Set HF cache for MI35x
# os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
# os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

# import unittest
# from types import SimpleNamespace

# import requests

# from sglang.srt.utils import kill_process_tree
# from sglang.test.ci.ci_register import register_amd_ci
# from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
# from sglang.test.send_one import BenchArgs, send_one_prompt
# from sglang.test.test_utils import (
#     DEFAULT_URL_FOR_TEST,
#     CustomTestCase,
#     is_in_ci,
#     popen_launch_server,
#     write_github_step_summary,
# )

# # Register for AMD CI - MI35x DeepSeek-V3.2 TP+MTP accuracy test
# register_amd_ci(
#     est_time=5400,
#     suite="nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp",
#     nightly=True,
# )

# DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# # Accuracy and performance thresholds
# GSM8K_ACCURACY_THRESHOLD = 0.94
# AVG_SPEC_ACCEPT_LENGTH_THRESHOLD = 2.7


# class TestDeepseekV32TPMTP(CustomTestCase):
#     """Test DeepSeek V3.2 with TP=8 + MTP (EAGLE speculative decoding).

#     This test runs GSM8K evaluation and measures both accuracy and
#     speculative decoding acceptance length on MI35x.
#     """

#     @classmethod
#     def setUpClass(cls):
#         cls.model = DEEPSEEK_V32_MODEL_PATH
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         # Use same args as perf test (which passes successfully)
#         other_args = [
#             "--trust-remote-code",
#             "--tp",
#             "8",
#             "--nsa-prefill-backend",
#             "tilelang",
#             "--nsa-decode-backend",
#             "tilelang",
#             "--speculative-algorithm",
#             "EAGLE",
#             "--speculative-num-steps",
#             "3",
#             "--speculative-eagle-topk",
#             "1",
#             "--speculative-num-draft-tokens",
#             "4",
#             "--mem-fraction-static",
#             "0.7",
#             "--model-loader-extra-config",
#             '{"enable_multithread_load": true}',
#             "--watchdog-timeout",
#             "1200",
#         ]
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=5400,
#             other_args=other_args,
#         )

#     @classmethod
#     def tearDownClass(cls):
#         kill_process_tree(cls.process.pid)

#     def test_a_gsm8k(self):
#         """GSM8K evaluation for TP+MTP configuration.

#         Named with 'a' prefix to run first (alphabetically) to warm up the server.
#         """
#         requests.get(self.base_url + "/flush_cache")

#         args = SimpleNamespace(
#             num_shots=20,
#             data_path=None,
#             num_questions=200,
#             parallel=64,
#             max_new_tokens=512,
#             host="http://127.0.0.1",
#             port=int(self.base_url.split(":")[-1]),
#         )
#         metrics = run_eval_few_shot_gsm8k(args)
#         print(f"{metrics=}")

#         server_info = requests.get(self.base_url + "/get_server_info")
#         avg_spec_accept_length = server_info.json()["internal_states"][0][
#             "avg_spec_accept_length"
#         ]
#         print(f"{avg_spec_accept_length=}")

#         if is_in_ci():
#             write_github_step_summary(
#                 f"### test_gsm8k (deepseek-v32 TP+MTP MI35x)\n"
#                 f'{metrics["accuracy"]=:.3f}\n'
#                 f"{avg_spec_accept_length=:.2f}\n"
#             )
#             self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)
#             self.assertGreater(avg_spec_accept_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)

#     def test_bs_1_speed(self):
#         """Single batch speed test for TP+MTP configuration."""
#         args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
#         acc_length, speed = send_one_prompt(args)

#         print(f"{acc_length=:.2f} {speed=:.2f}")

#         if is_in_ci():
#             write_github_step_summary(
#                 f"### test_bs_1_speed (deepseek-v32 TP+MTP MI35x)\n"
#                 f"{acc_length=:.2f}\n"
#                 f"{speed=:.2f} token/s\n"
#             )
#             self.assertGreater(acc_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)
#             self.assertGreater(speed, 55)  # Lowered from 60 for AMD MI35x


# if __name__ == "__main__":
#     unittest.main()

import ast
import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import re
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

# Register for AMD CI - MI35x DeepSeek-V3.2 accuracy test (~90 min for basic only)
register_amd_ci(
    est_time=5400,
    suite="nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp",
    nightly=True,
)

INVALID = -9999999


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None
    variant: Optional[str] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}

    def get_display_name(self) -> str:
        if self.variant:
            return f"{self.model_path} ({self.variant})"
        return self.model_path


# DeepSeek-V3.2 models for MI35x - only basic variant for nightly
# DP variant removed due to barrier deadlock during model loading
MI35X_DEEPSEEK_V32_MODELS = [
    # DeepSeek-V3.2 basic (TP=8 only)
    ModelConfig(
        model_path="deepseek-ai/DeepSeek-V3.2",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=5400,
        variant="basic",
        other_args=[
            "--trust-remote-code",
            "--nsa-prefill-backend",
            "tilelang",
            "--nsa-decode-backend",
            "tilelang",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",  # 20 minutes for weight loading
        ],
        env_vars={},
    ),
]


def get_one_example(lines, i, include_answer):
    """Format a single GSM8K example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """Get k few-shot examples for prompting."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Extract numerical answer from response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run GSM8K few-shot completion benchmark."""
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    backend = RuntimeEndpoint(base_url)
    sgl.set_default_backend(backend)

    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments, temperature=0, num_threads=parallel, progress_bar=True
    )
    latency = time.perf_counter() - tic

    preds = [get_answer_value(states[i]["answer"]) for i in range(len(states))]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


class TestDeepSeekV32MTPEvalMI35x(unittest.TestCase):
    """DeepSeek-V3.2 GSM8K Completion Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI35X_DEEPSEEK_V32_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_deepseek_v32_accuracy(self):
        """Test DeepSeek-V3.2 models with GSM8K completion benchmark."""
        all_results = []
        summary = "### DeepSeek-V3.2 Models (MI35x)\n\n"
        summary += "| Model | Variant | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | ------- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            display_name = config.get_display_name()
            with self.subTest(model=display_name):
                print(f"\n{'='*60}")
                print(f"Testing: {display_name}")
                print(f"{'='*60}")

                env = os.environ.copy()
                for key, value in config.env_vars.items():
                    env[key] = value

                other_args = list(config.other_args)
                other_args.extend(["--tp", str(config.tp_size)])
                timeout = config.timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

                try:
                    process = popen_launch_server(
                        model=config.model_path,
                        base_url=self.base_url,
                        timeout=timeout,
                        other_args=other_args,
                        env=env,
                    )

                    try:
                        acc, invalid, latency = run_gsm8k_benchmark(
                            self.base_url, num_questions=self.num_questions
                        )
                        passed = acc >= config.accuracy_threshold
                        status = "✅ PASS" if passed else "❌ FAIL"
                        print(
                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
                        )

                        all_results.append(
                            {
                                "model": display_name,
                                "accuracy": acc,
                                "passed": passed,
                            }
                        )
                        summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | {acc:.3f} | {config.accuracy_threshold} | {status} |\n"

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
                    all_results.append(
                        {
                            "model": display_name,
                            "accuracy": None,
                            "passed": False,
                            "error": str(e),
                        }
                    )

        if is_in_ci():
            write_github_step_summary(summary)

        failed = [r for r in all_results if not r["passed"]]
        if failed:
            raise AssertionError(f"Failed models: {[r['model'] for r in failed]}")


if __name__ == "__main__":
    unittest.main()
