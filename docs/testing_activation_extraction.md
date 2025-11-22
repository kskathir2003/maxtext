# Testing Activation Extraction in MaxText

This guide provides step-by-step instructions for building, testing, and validating the activation extraction feature for Qwen3 models.

## Quick Start

### 1. Setup Environment

First, clone MaxText and install dependencies:

```bash
# Clone the repository
git clone https://github.com/kskathir2003/maxtext.git
cd maxtext

# Create virtual environment
python3.12 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate

# Install dependencies (TPU)
bash tools/setup/setup.sh DEVICE=tpu
# Or for GPU:
# bash tools/setup/setup.sh DEVICE=gpu

# Alternatively, use pip:
pip install uv
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps
```

### 2. Run Unit Tests

Test the activation extraction implementation:

```bash
# Run activation extraction tests
python -m unittest tests.activation_extraction_test -v

# Or run all tests
python -m pytest tests/ -v
```

**Expected output**: All 6 test cases should pass:
- `test_capture_disabled_by_default`
- `test_qwen3_decoder_layer_has_layer_index`
- `test_should_capture_logic`
- `test_activation_capture_non_scanned`
- `test_moe_layer_has_layer_index`
- `test_no_overhead_when_disabled`

### 3. Run Example Script

Test the example activation collector:

```bash
# Run the example script
python src/MaxText/examples/extract_activations_qwen3.py \
  --config=src/MaxText/configs/models/qwen3-0.6b.yml \
  --output_dir=/tmp/activations \
  --num_samples=100 \
  --capture_layers=0,6,12 \
  --capture_types=post_mlp,post_attn
```

**Expected output**: The script should:
1. Load the Qwen3-0.6B config
2. Simulate forward passes with activation capture
3. Save activation tensors to `/tmp/activations/`
4. Print summary statistics

### 4. Test with Real Training

Run a minimal training job with activation capture enabled:

```bash
# Set up GCS bucket (replace with your bucket)
export BASE_OUTPUT_DIR=gs://your-bucket/maxtext-test
export RUN_NAME=test_activation_capture_$(date +%Y%m%d_%H%M%S)

# Run short training with activation capture
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=$RUN_NAME \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=10 \
  capture_activations=true \
  activation_capture_layers=[0,6,12] \
  activation_capture_types=['post_mlp','post_attn'] \
  per_device_batch_size=1 \
  max_target_length=128
```

**Note**: This will train for only 10 steps with synthetic data to verify the feature works end-to-end.

### 5. Validate Activation Capture

To verify activations are being captured correctly, you can modify the training script to print activation shapes:

```python
# Add to your training code
if config.capture_activations:
  variables = model.apply(
      {'params': params},
      decoder_input_tokens=batch['inputs'],
      decoder_positions=batch['positions'],
      capture_intermediates=lambda m, method: method == "__call__",
      mutable=['activations']
  )
  
  if 'intermediates' in variables and 'activations' in variables['intermediates']:
    activations = variables['intermediates']['activations']
    print(f"Captured activations: {list(activations.keys())}")
    for key, value in activations.items():
      print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
```

## Testing Different Configurations

### Test 1: Capture All Layers and Types

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=test_all_activations \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=5 \
  capture_activations=true \
  activation_capture_layers=[] \
  activation_capture_types=[] \
  per_device_batch_size=1
```

**Expected**: Captures all activation types from all layers

### Test 2: Selective Layer Capture

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=test_selective_layers \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=5 \
  capture_activations=true \
  activation_capture_layers=[0,11,23] \
  activation_capture_types=['post_mlp'] \
  per_device_batch_size=1
```

**Expected**: Only captures `post_mlp` activations from layers 0, 11, and 23

### Test 3: Test with Scanned Layers

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=test_scanned_layers \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=5 \
  capture_activations=true \
  activation_capture_layers=[0,12] \
  activation_capture_types=['post_mlp','output'] \
  scan_layers=true \
  per_device_batch_size=1
```

**Expected**: Activations stacked along first dimension: `{post_mlp: [num_layers, B, L, D]}`

### Test 4: MoE Model Testing

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-30b-a3b \
  run_name=test_moe_activations \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=5 \
  capture_activations=true \
  activation_capture_layers=[0,5,10] \
  activation_capture_types=['post_mlp'] \
  per_device_batch_size=1
```

**Expected**: Works with MoE layers (Qwen3MoeDecoderLayer)

### Test 5: Zero Overhead When Disabled

```bash
# Run WITHOUT activation capture
time python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=test_disabled \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=100 \
  capture_activations=false \
  per_device_batch_size=2

# Compare with ENABLED (should be < 2% slower)
time python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=qwen3-0.6b \
  run_name=test_enabled \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=100 \
  capture_activations=true \
  activation_capture_layers=[0,12] \
  activation_capture_types=['post_mlp'] \
  per_device_batch_size=2
```

**Expected**: < 2% overhead when enabled with selective capture

## Integration Testing

### Test with Real Qwen3 Checkpoint

```bash
# 1. Download and convert Qwen3-4B checkpoint
export HF_TOKEN=your_huggingface_token
export MODEL_BUCKET=gs://your-bucket/qwen3
export MODEL_NAME=qwen3-4b

python -m MaxText.utils.ckpt_conversion.to_maxtext \
  src/MaxText/configs/base.yml \
  model_name=${MODEL_NAME} \
  hf_access_token=${HF_TOKEN} \
  base_output_directory=${MODEL_BUCKET}/4b/unscanned \
  scan_layers=false

# 2. Run inference with activation capture
python3 -m MaxText.decode src/MaxText/configs/base.yml \
  model_name=qwen3-4b \
  tokenizer_path=src/MaxText/assets/qwen3-tokenizer \
  load_parameters_path=${MODEL_BUCKET}/4b/unscanned/0/items \
  per_device_batch_size=1 \
  run_name=test_decode_activations \
  max_prefill_predict_length=32 \
  max_target_length=64 \
  dataset_type=synthetic \
  steps=5 \
  scan_layers=false \
  capture_activations=true \
  activation_capture_layers=[0,6,12] \
  activation_capture_types=['post_mlp','post_attn'] \
  prompt="I love to"
```

## Performance Benchmarking

To measure the overhead of activation capture:

```bash
# Create a benchmark script
cat > /tmp/benchmark_activations.sh << 'EOF'
#!/bin/bash

BASE_OUTPUT_DIR=gs://your-bucket/benchmark
MODEL_NAME=qwen3-0.6b
STEPS=100

echo "=== Baseline (disabled) ==="
time python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=$MODEL_NAME \
  run_name=bench_baseline \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=$STEPS \
  capture_activations=false \
  per_device_batch_size=4

echo "=== Selective capture ==="
time python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=$MODEL_NAME \
  run_name=bench_selective \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=$STEPS \
  capture_activations=true \
  activation_capture_layers=[0,12] \
  activation_capture_types=['post_mlp'] \
  per_device_batch_size=4

echo "=== Full capture ==="
time python3 -m MaxText.train src/MaxText/configs/base.yml \
  model_name=$MODEL_NAME \
  run_name=bench_full \
  base_output_directory=$BASE_OUTPUT_DIR \
  dataset_type=synthetic \
  steps=$STEPS \
  capture_activations=true \
  activation_capture_layers=[] \
  activation_capture_types=[] \
  per_device_batch_size=4
EOF

chmod +x /tmp/benchmark_activations.sh
bash /tmp/benchmark_activations.sh
```

## Troubleshooting

### Issue: Tests won't run - JAX not installed

**Solution**: Make sure you've installed dependencies:
```bash
source ~/venv-maxtext/bin/activate
pip install uv
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps
```

### Issue: Activations not appearing in output

**Solution**: Ensure you're passing `mutable=['activations']` in `model.apply()`:
```python
variables = model.apply(
    {'params': params},
    ...,
    mutable=['activations']  # Required!
)
```

### Issue: Out of memory during capture

**Solutions**:
1. Reduce number of captured layers
2. Use selective activation types
3. Reduce batch size
4. Enable periodic sampling (capture every N steps)

### Issue: Wrong activation shapes

For scanned layers, activations have shape `[num_layers, batch, seq_len, hidden_dim]`. 
Use indexing: `activations['post_mlp'][12]` to get layer 12.

## Continuous Integration

The activation extraction feature is tested in CI via:
- Unit tests: `tests/activation_extraction_test.py`
- End-to-end tests: Will be added to `.github/workflows/`

To run locally what CI runs:

```bash
# Run unit tests
python -m pytest tests/activation_extraction_test.py -v

# Check code style
python -m py_compile src/MaxText/layers/qwen3.py
python -m py_compile src/MaxText/layers/decoders.py
python -m py_compile src/MaxText/examples/extract_activations_qwen3.py
```

## Next Steps

After validating the feature works:
1. Use captured activations for SAE training
2. Analyze activation patterns for interpretability
3. Visualize learned features
4. Export activations for external tools

See `docs/activation_extraction.md` for detailed usage guide and advanced patterns.
