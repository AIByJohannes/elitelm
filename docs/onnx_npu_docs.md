# NPU support in snapdragon qnn package for Hexagon NPU llm inference: is it supported in python or only in c++ as of today?

Short answer: NPU-backed LLM inference with Snapdragon’s QNN stack is available from Python via ONNX Runtime’s QNN Execution Provider; the native QNN runtime itself exposes a C-style API typically used from C/C++ as the primary interface today. Community/Qualcomm wrappers like QAI AppBuilder also offer Python wheels on Windows on Snapdragon and Linux, but they build on the underlying C/C++ runtime and are not the core SDK API.[^1_1][^1_2][^1_3]

### Python support

- ONNX Runtime provides a QNN Execution Provider with a Python package (onnxruntime-qnn) that runs quantized models on the HTP backend, which is the Hexagon NPU, and includes Python examples for creating a session with QNN on Windows ARM64 devices.[^1_1]
- The QNN EP targets Android and Windows, but prebuilt Python wheels are currently provided for Windows ARM64; Android usage typically involves building from source or using native interfaces, per the QNN EP documentation scope and build notes.[^1_1]


### Native SDK reality

- The QNN API is explicitly described as a C-style API for portability, which means the official runtime interface is in C/C++ rather than a first-class Python API for execution graphs.[^1_2]
- Python appears in the SDK setup and tooling context, but execution-level Python access is generally mediated through higher layers like ONNX Runtime or helper wrappers, not through a direct, official Python runtime API in the QNN SDK.[^1_2]


### LLM-specific constraints

- Running on HTP requires a quantized model; the QNN EP Python guide states HTP supports only quantized models and also highlights that dynamic shapes are not supported, which affects typical LLM dynamic sequence lengths unless fixed ahead of time.[^1_1]
- ONNX Runtime GenAI guidance exists for building model assets and QNN binaries for Snapdragon NPU targets, indicating an expected workflow for deploying LLM assets onto the NPU through the QNN stack.[^1_4]


### Alternate Python paths

- Qualcomm’s QAI AppBuilder publishes Python wheels that wrap QNN for Windows on Snapdragon (and Linux/aarch64), and includes an example “Genie” chat-style usage that demonstrates LLM-oriented flows from Python, while still relying on the underlying QNN runtime components.[^1_3]
- External tutorials and blogs also demonstrate Python-based NPU inference on Snapdragon devices using ONNX Runtime with QNN EP, reinforcing practical Python support for NPU-backed inference workflows today.[^1_5]


### Bottom line

- For Python today, the supported path is to use ONNX Runtime’s QNN Execution Provider (with quantized models) to target the Hexagon NPU (HTP), not a native QNN Python runtime API.[^1_2][^1_1]
- For maximal control and portability, C/C++ remains the canonical interface to QNN, while Python workflows should go through ONNX Runtime QNN EP or wrappers like QAI AppBuilder depending on platform and deployment needs.[^1_3][^1_2]
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_6][^1_7][^1_8][^1_9]</span>

<div style="text-align: center">⁂</div>

[^1_1]: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

[^1_2]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/api_overview.html?product=1601111740009302

[^1_3]: https://github.com/quic/ai-engine-direct-helper/releases

[^1_4]: https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html

[^1_5]: https://techcommunity.microsoft.com/blog/surfaceitpro/unlocking-the-power-of-npu-on-surface-our-“hello-world”-journey/4149473

[^1_6]: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/qnn.html

[^1_7]: https://docs.pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html

[^1_8]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html

[^1_9]: https://docs.pytorch.org/executorch/0.3/build-run-qualcomm-ai-engine-direct-backend.html

[^1_10]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html

[^1_11]: https://github.com/quic/ai-engine-direct-helper

[^1_12]: https://docs.qualcomm.com/bundle/publicresource/topics/80-70015-15B/qnn-setup.html

[^1_13]: https://github.com/MollySophia/rwkv-qualcomm

[^1_14]: https://github.com/pytorch/executorch/issues/9474

[^1_15]: https://mysupport.qualcomm.com/supportforums/s/question/0D5dK000002smEmSAI/what-are-the-differences-between-the-ai-engine-direct-sdk-the-neural-processing-sdk-and-the-hexagon-npu-sdk

[^1_16]: https://github.com/quic/ai-hub-models

[^1_17]: https://developer.advantech.com/EdgeSync/Containers/Environment/Qualcomm

[^1_18]: https://pkbullock.com/blog/2024/running-models-using-npu-with-copilot-pc

[^1_19]: https://ai.google.dev/edge/litert/android/npu/qualcomm

[^1_20]: https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.1.1/ai-developer-workflow/

[^1_21]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/windows_setup.html

[^1_22]: https://www.reddit.com/r/Surface/comments/1dojw6v/what_is_the_point_of_the_npu_in_the_snapdragon/

[^1_23]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html

[^1_24]: https://mysupport.qualcomm.com/supportforums/s/question/0D5dK000002duUQSAY/sample-cpp-application-for-running-inference-using-qnn-sdk

[^1_25]: https://github.com/quic/qidk

[^1_26]: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-100/setup.html?product=1601111740062489

[^1_27]: https://quic.github.io/cloud-ai-sdk-pages/latest/Python-API/qaic/qaic/

[^1_28]: https://github.com/pytorch/executorch/issues/13766

[^1_29]: https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Scripts/qnn_setup.ps1

[^1_30]: https://onnxruntime.ai/docs/install/


---

# find a specific code example of how to run inference on npu for an llm using one of these python APIs

Direct answer: Here are two concrete Python code examples that run LLM inference on Snapdragon’s Hexagon NPU via supported Python APIs: one using ONNX Runtime GenAI configured with the QNN Execution Provider, and another using Qualcomm’s QAI AppBuilder “Genie” wrapper for QNN.[^2_1][^2_2][^2_3]

### ONNX Runtime GenAI + QNN EP

This example loads an ORT-GenAI LLM with a genai_config.json, selects the QNN Execution Provider, points it to the HTP backend library, and streams generated tokens; it requires onnxruntime-genai and a QNN-enabled ONNX Runtime (onnxruntime-qnn) plus Snapdragon-formatted model assets built per the Snapdragon guide.[^2_4][^2_3][^2_1]

```python
# Minimal streaming text-generation on Snapdragon NPU via ORT-GenAI + QNN EP.
# Requires: pip install onnxruntime-genai and a QNN-enabled ORT (e.g., onnxruntime-qnn),
# and Snapdragon-ready assets (genai_config.json and binaries).

import os
import onnxruntime_genai as og
import numpy as np

# Optional: verify QNN is available in this build/runtime
print("QNN available:", og.is_qnn_available())

# Load config generated for the LLM (from Snapdragon build steps)
config = og.Config("path/to/genai_config.json")

# Force use of QNN EP (Hexagon NPU/HTP) instead of whatever the config lists
config.clear_providers()
config.append_provider("QNNExecutionProvider")

# Point the EP to the HTP backend library from the QNN SDK:
#   Windows ARM64: "C:\\Qualcomm\\AIStack\\QAIRT\\<ver>\\lib\\arm64x-windows-msvc\\QnnHtp.dll"
#   Linux/Android: "/path/to/libQnnHtp.so"
config.set_provider_option("backend_path", "path/to/QnnHtp.dll_or_libQnnHtp.so")

# Load model
model = og.Model(config)

# Tokenize a prompt
tokenizer = og.Tokenizer(model)
prompt = "You are a helpful assistant. Explain quantization in simple terms:"
input_ids = tokenizer.encode(prompt)

# Set generation parameters
params = og.GeneratorParams(model)
params.set_model_input("input_ids", np.array(input_ids, dtype=np.int32))
params.set_search_options(temperature=0.7, top_p=0.9)

# Stream tokens
generator = og.Generator(model, params)
stream = tokenizer.create_stream()

while not generator.is_done():
    generator.generate_next_token()
    next_tokens = generator.get_next_tokens()
    if next_tokens.size > 0:
        print(stream.decode(int(next_tokens[-1])), end="", flush=True)

print()
```

The provider setting and backend_path option correspond to the QNN EP’s documented configuration for running on the HTP backend, and ORT-GenAI’s Python API exposes the Config/Model/Tokenizer/Generator classes used above.[^2_3][^2_1]

<div style="text-align: center">⁂</div>

[^2_1]: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

[^2_2]: https://github.com/quic/ai-engine-direct-helper/releases

[^2_3]: https://onnxruntime.ai/docs/genai/api/python.html

[^2_4]: https://onnxruntime.ai/docs/genai/tutorials/snapdragon.html

[^2_5]: https://onnxruntime.ai/docs/execution-providers/

[^2_6]: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/ort-qnn-ep.html

[^2_7]: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers

[^2_8]: https://onnxruntime.ai/docs/get-started/with-python.html

[^2_9]: https://github.com/microsoft/onnxruntime

[^2_10]: https://github.com/quic/ai-engine-direct-helper

[^2_11]: https://fs-eire.github.io/onnxruntime/docs/build/eps.html

[^2_12]: https://www.youtube.com/watch?v=So9GyWTSyAc

[^2_13]: https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html

[^2_14]: https://onnxruntime.ai/docs/install/

[^2_15]: https://databricks-sdk-py.readthedocs.io/en/stable/workspace/dashboards/genie.html

[^2_16]: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/hf-optimum-ort.html

[^2_17]: https://github.com/microsoft/onnxruntime/issues/23059

[^2_18]: https://www.ciscolive.com/c/dam/r/ciscolive/global-event/docs/2024/pdf/BRKATO-2108.pdf

[^2_19]: https://github.com/microsoft/onnxruntime-genai/issues/1417

[^2_20]: https://pypi.org/project/onnxruntime-qnn/

[^2_21]: https://www.youtube.com/watch?v=tNq9MP9wlBg

[^2_22]: https://github.com/microsoft/onnxruntime/issues/21203

[^2_23]: https://onnxruntime.ai/docs/genai/api/java.html

[^2_24]: https://devblogs.microsoft.com/semantic-kernel/onnx-genai-connector-for-python-experimental/

[^2_25]: https://docs.pytorch.org/executorch/0.3/build-run-qualcomm-ai-engine-direct-backend.html

[^2_26]: https://github.com/microsoft/onnxruntime-genai/issues/1728

[^2_27]: https://github.com/microsoft/onnxruntime-genai

[^2_28]: https://pkbullock.com/blog/2024/running-models-using-npu-with-copilot-pc

[^2_29]: https://github.com/microsoft/onnxruntime-genai/discussions/660

[^2_30]: https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py

[^2_31]: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/ort-qnn-ep.html?product=1601111740057789

[^2_32]: https://onnxruntime.ai/docs/genai/reference/config.html

[^2_33]: https://github.com/microsoft/onnxruntime-genai/issues/1098

[^2_34]: https://onnxruntime.ai/docs/genai/api/cpp.html

[^2_35]: https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html

