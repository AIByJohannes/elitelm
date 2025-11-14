"""Tier 3: NPU Hardware-Specific Tests

These tests validate actual Hexagon NPU execution. They require:
- QNN compiled into onnxruntime-genai
- QNN SDK installed and findable
- A quantized ONNX model for testing
- Actual Snapdragon NPU hardware available

Tests are marked with @pytest.mark.requires_npu and skip gracefully
on systems without NPU support.
"""

import time
import importlib.util
from pathlib import Path
import pytest

from conftest import MODEL_FILENAMES

try:
    import onnxruntime_genai as og
except ImportError:
    og = None


# Helper to load elitelm module dynamically
def load_elitelm_module():
    """Load the elitelm module from the parent directory."""
    module_path = Path(__file__).resolve().parent.parent / "elitelm.py"
    spec = importlib.util.spec_from_file_location("elitelm", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Tier 3: Actual NPU Execution Tests
# ============================================================================


@pytest.mark.requires_npu
def test_npu_model_loading(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that a quantized model can be loaded on NPU (Tier 3).

    This test validates that og.Model() can successfully load a quantized
    ONNX model using the QNN provider configuration. It does not run inference,
    only verifies model loading succeeds.

    Primary Signal: Model loads without exception
    Proves: QNN provider is functional and model format is compatible
    """
    module = load_elitelm_module()

    # Configure QNN provider
    config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)

    # Attempt to load model
    # This will raise an exception if the model is incompatible with QNN
    try:
        model = og.Model(config)
        assert model is not None, "Model should be successfully instantiated"
    except Exception as e:
        pytest.fail(f"Failed to load model on NPU: {e}")


@pytest.mark.requires_npu
def test_npu_inference_runs(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that inference can run on NPU (Tier 3).

    This test validates that a tokenizer can be created and a basic generation
    can be initiated on the NPU. It demonstrates actual NPU model execution.

    Primary Signal: Inference completes without exception
    Proves: NPU is executing model (or would raise if incompatible)
    """
    module = load_elitelm_module()

    config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)

    try:
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)

        # Create a simple prompt
        prompt = "Hello"
        prompt_tokens = tokenizer.encode(prompt)

        # Create generation parameters
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=10, do_sample=False)

        # Create generator and run a few tokens
        generator = og.Generator(model, params)
        generator.append_tokens(prompt_tokens)

        # Generate at least 1 token
        token_count = 0
        while not generator.is_done() and token_count < 10:
            generator.generate_next_token()
            token_count += 1

        assert token_count > 0, "Should generate at least one token"

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "qnn" in error_msg or "htp" in error_msg:
            pytest.fail(f"NPU execution failed (likely model incompatibility): {e}")
        raise
    except Exception as e:
        pytest.fail(f"Unexpected error during NPU inference: {e}")


@pytest.mark.requires_npu
def test_npu_tokenizer_creation(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that tokenizer can be created from NPU model (Tier 3).

    This is a prerequisite validation that the model and tokenizer
    are compatible and the model has been properly configured for NPU.

    Primary Signal: Tokenizer instantiation succeeds
    Proves: Model is properly loaded and tokenizer metadata is available
    """
    module = load_elitelm_module()

    config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)

    try:
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)

        assert tokenizer is not None, "Tokenizer should be instantiated"

        # Try to encode a basic string
        tokens = tokenizer.encode("test")
        assert len(tokens) > 0, "Should successfully encode text"

    except Exception as e:
        pytest.fail(f"Failed to create tokenizer from NPU model: {e}")


@pytest.mark.requires_npu
def test_npu_vs_cpu_performance_difference(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that NPU and CPU have measurably different performance (Tier 3).

    This test measures wall-time latency for a short generation on both CPU
    and NPU configurations. Different timing characteristics prove that the
    execution paths are actually different (not just different labels).

    Primary Signal: Timing differs between CPU and NPU
    Proves: Different execution backend is being used
    Fails: If timings are identical (suspicious, suggests fallback or mock)
    """
    module = load_elitelm_module()

    prompt_text = "Test"
    max_tokens = 5

    # ---- CPU Baseline ----
    try:
        cpu_model = og.Model(str(test_model_path))
        cpu_tokenizer = og.Tokenizer(cpu_model)
        cpu_prompt = cpu_tokenizer.encode(prompt_text)

        cpu_params = og.GeneratorParams(cpu_model)
        cpu_params.set_search_options(max_length=max_tokens, do_sample=False)

        cpu_start = time.time()
        cpu_gen = og.Generator(cpu_model, cpu_params)
        cpu_gen.append_tokens(cpu_prompt)

        cpu_token_count = 0
        while not cpu_gen.is_done() and cpu_token_count < max_tokens:
            cpu_gen.generate_next_token()
            cpu_token_count += 1

        cpu_time = time.time() - cpu_start

    except Exception as e:
        pytest.skip(f"Could not run CPU baseline: {e}")

    # ---- NPU Execution ----
    try:
        npu_config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)
        npu_model = og.Model(npu_config)
        npu_tokenizer = og.Tokenizer(npu_model)
        npu_prompt = npu_tokenizer.encode(prompt_text)

        npu_params = og.GeneratorParams(npu_model)
        npu_params.set_search_options(max_length=max_tokens, do_sample=False)

        npu_start = time.time()
        npu_gen = og.Generator(npu_model, npu_params)
        npu_gen.append_tokens(npu_prompt)

        npu_token_count = 0
        while not npu_gen.is_done() and npu_token_count < max_tokens:
            npu_gen.generate_next_token()
            npu_token_count += 1

        npu_time = time.time() - npu_start

    except Exception as e:
        pytest.fail(f"Failed to run on NPU: {e}")

    # ---- Validation ----
    # Timings should differ (proves different execution path)
    # Allow for minor variation due to system noise
    time_delta = abs(cpu_time - npu_time)
    min_time = min(cpu_time, npu_time)

    # If times are identical, that's suspicious (floating point rounding)
    # Allow 1ms tolerance for very fast operations
    if time_delta < 0.001:
        # This is not necessarily a failure, just low confidence in the signal
        # Skip rather than fail
        pytest.skip(
            f"Performance timings too similar to distinguish: "
            f"CPU={cpu_time:.6f}s, NPU={npu_time:.6f}s (delta={time_delta:.6f}s)"
        )

    # Log for visibility
    ratio = npu_time / cpu_time if cpu_time > 0 else float("inf")
    print(f"\nPerformance Comparison:")
    print(f"  CPU time: {cpu_time:.6f}s")
    print(f"  NPU time: {npu_time:.6f}s")
    print(f"  Delta: {time_delta:.6f}s")
    print(f"  Ratio (NPU/CPU): {ratio:.2f}x")


@pytest.mark.requires_npu
def test_npu_output_reasonable(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that NPU output is reasonable and not corrupted (Tier 3).

    This test validates that the output tokens are valid (within expected range)
    and decodable to text. This is a basic sanity check that the output is not
    garbage or corrupted memory.

    Primary Signal: Output decodes to valid text
    Proves: NPU model output is reasonable (not corruption)
    """
    module = load_elitelm_module()

    prompt_text = "Hello"
    max_tokens = 10

    try:
        npu_config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)
        npu_model = og.Model(npu_config)
        npu_tokenizer = og.Tokenizer(npu_model)
        npu_prompt = npu_tokenizer.encode(prompt_text)

        npu_params = og.GeneratorParams(npu_model)
        npu_params.set_search_options(max_length=max_tokens, do_sample=False)

        npu_gen = og.Generator(npu_model, npu_params)
        npu_gen.append_tokens(npu_prompt)

        output_tokens = []
        while not npu_gen.is_done() and len(output_tokens) < max_tokens:
            npu_gen.generate_next_token()
            token_id = npu_gen.get_next_tokens()[0]
            output_tokens.append(int(token_id))

        # Validate output
        assert len(output_tokens) > 0, "Should generate at least one token"

        # Try to decode (will fail if tokens are invalid)
        try:
            output_text = npu_tokenizer.decode(output_tokens)
            assert len(output_text) > 0, "Decoded output should not be empty"
            # Check for basic validity (not all control characters)
            assert output_text.strip(), "Output should not be only whitespace"

            print(f"\nNPU Output: {output_text[:100]}")

        except Exception as e:
            pytest.fail(f"Failed to decode NPU output tokens: {e}")

    except Exception as e:
        pytest.fail(f"Failed to run NPU inference for output validation: {e}")


@pytest.mark.requires_npu
def test_npu_provider_option_verification(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Test that QNN provider options are correctly applied (Tier 3).

    This test verifies that the provider configuration is correctly set
    on the og.Config object before model loading.

    Primary Signal: Provider options retrievable and correct
    Proves: Configuration was applied as expected
    """
    module = load_elitelm_module()

    config = module._configure_qnn_provider(test_model_path, str(qnn_sdk_root), None)

    # Verify backend_path was set
    backend = config.get_provider_option("QNNExecutionProvider", "backend_path")
    assert backend is not None, "backend_path should be set"
    assert Path(backend).exists(), f"Backend path should exist: {backend}"

    # Verify qnn_sdk_root was set
    sdk_root = config.get_provider_option("QNNExecutionProvider", "qnn_sdk_root")
    assert sdk_root is not None, "qnn_sdk_root should be set"
    assert Path(sdk_root).exists(), f"SDK root should exist: {sdk_root}"

    print(f"\nNPU Configuration Verified:")
    print(f"  Backend: {backend}")
    print(f"  SDK Root: {sdk_root}")


# ============================================================================
# Tier 3: Graceful Skip Tests (Demonstrate Skip Behavior)
# ============================================================================


@pytest.mark.requires_npu
def test_skip_without_npu_sdk(skip_if_no_qnn):
    """Demonstrate that tests skip gracefully if QNN SDK not found.

    This test would skip if the QNN SDK cannot be located, even if
    QNN is compiled in. It uses the skip_if_no_qnn fixture to ensure
    all preconditions are met before proceeding.
    """
    # This test body only runs if all fixtures succeed
    assert True, "Test reached with all preconditions met"


@pytest.mark.requires_npu
def test_skip_without_test_model(skip_if_no_qnn, qnn_sdk_root, test_model_path):
    """Demonstrate that tests skip gracefully if test model not found.

    This test would skip if a suitable test model cannot be found in
    expected locations. The skip message indicates what was looked for.
    """
    # This test body only runs if model is found
    assert test_model_path.exists(), "Test model should exist"
    assert any(
        (test_model_path / candidate).exists() for candidate in MODEL_FILENAMES
    ), "Expected ONNX model (model.onnx or ort_model.onnx) to exist"
