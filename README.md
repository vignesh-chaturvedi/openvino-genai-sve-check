# openvino-genai-sve-check

Checks whether the PagedAttention based features of openvino.genai 2026.4.0 (continuous batching, speculative decoding, prompt lookup) work on an ARM64 CPU that has SVE.

openvino.genai decides this with a compile time check in `src/cpp/src/utils.cpp`:

```cpp
#if defined(OPENVINO_ARCH_X86_64) || (defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE))
```

OpenVINO core selects its SVE PagedAttention kernel at runtime, so this repo runs the same cases on an SVE2 runner (`ubuntu-24.04-arm`, Azure Cobalt 100) and on an x86_64 runner as a control.

`check.py` builds each pipeline on CPU and generates 8 tokens with `OpenVINO/Qwen3-0.6B-int4-ov`. `ContinuousBatchingPipeline` is included because it uses PagedAttention without going through the openvino.genai check.

Results are in the job summary of each run in the Actions tab.

`version-check.yml` runs the same checks on the SVE2 runner for every openvino.genai release from 2026.0.0 (before the `HAVE_SVE` condition was added) to 2026.4.1. It is started manually from the Actions tab.

Run locally:

```
pip install openvino==2026.4.0 openvino-genai==2026.4.0.0 openvino-tokenizers==2026.4.0.0 huggingface_hub
hf download OpenVINO/Qwen3-0.6B-int4-ov --local-dir model
python check.py model
```
