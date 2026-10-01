"""Check which openvino.genai PagedAttention entry points work on this CPU.

Usage: python check.py <model_dir>

Every case builds a pipeline on CPU and generates 8 tokens. A case either
prints the generated text or the first line of the exception and the stage
(construct or generate) where it was raised.
"""

import os
import platform
import sys

import openvino as ov
import openvino_genai as ov_genai

PROMPT = "The sky is blue because"


def sve_flags():
    try:
        with open("/proc/cpuinfo") as f:
            words = set(f.read().split())
    except FileNotFoundError:
        return "no /proc/cpuinfo"
    found = [flag for flag in ("sve", "sve2") if flag in words]
    return ", ".join(found) if found else "none"


def generation_config(**fields):
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 8
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def llm_case(model_dir, config, **properties):
    def construct():
        return ov_genai.LLMPipeline(model_dir, "CPU", **properties)

    def generate(pipe):
        return str(pipe.generate(PROMPT, config))

    return construct, generate


def continuous_batching_case(model_dir):
    def construct():
        return ov_genai.ContinuousBatchingPipeline(model_dir, ov_genai.SchedulerConfig(), "CPU")

    def generate(pipe):
        # String prompts return GenerationResult, whose m_generation_ids are already decoded
        return pipe.generate([PROMPT], [generation_config()])[0].m_generation_ids[0]

    return construct, generate


def run(construct, generate):
    try:
        pipe = construct()
    except Exception as e:
        return "construct", str(e).strip().splitlines()[-1]
    try:
        return "ok", generate(pipe).strip().replace("\n", " ")
    except Exception as e:
        return "generate", str(e).strip().splitlines()[-1]


def main():
    model_dir = sys.argv[1]
    cpu = ov.Core().get_property("CPU", "FULL_DEVICE_NAME")

    cases = [
        ("LLMPipeline, default backend (control)", llm_case(model_dir, generation_config())),
        ("LLMPipeline, scheduler_config", llm_case(model_dir, generation_config(), scheduler_config=ov_genai.SchedulerConfig())),
        ("LLMPipeline, ATTENTION_BACKEND=PA", llm_case(model_dir, generation_config(), ATTENTION_BACKEND="PA")),
        ("LLMPipeline, prompt_lookup", llm_case(model_dir, generation_config(num_assistant_tokens=3, max_ngram_size=3), prompt_lookup=True)),
        ("LLMPipeline, draft_model", llm_case(model_dir, generation_config(num_assistant_tokens=3), draft_model=ov_genai.draft_model(model_dir, "CPU"))),
        ("ContinuousBatchingPipeline (no genai pre-check)", continuous_batching_case(model_dir)),
    ]

    lines = [
        f"- machine: `{platform.machine()}`, CPU: `{cpu}`",
        f"- SVE flags in /proc/cpuinfo: `{sve_flags()}`",
        f"- openvino `{ov.get_version()}`, openvino-genai `{ov_genai.__version__}`",
        "",
        "| Case | Result | Detail |",
        "|---|---|---|",
    ]
    for name, (construct, generate) in cases:
        stage, detail = run(construct, generate)
        result = "works" if stage == "ok" else f"fails at {stage}"
        lines.append(f"| {name} | {result} | `{detail[:200]}` |")
        print(f"{name}: {result}: {detail}", flush=True)

    report = "\n".join(lines)
    print("\n" + report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## {platform.machine()}\n\n{report}\n")


if __name__ == "__main__":
    main()
