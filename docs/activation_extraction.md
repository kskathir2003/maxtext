# Activation Extraction for SAE Training and Mechanistic Interpretability

This guide explains how to use MaxText's activation extraction feature for Sparse Autoencoder (SAE) training and mechanistic interpretability research on Qwen3-based models.

## Overview

MaxText supports capturing intermediate activations during forward passes using Flax's `sow` mechanism. This feature enables:
- Training Sparse Autoencoders (SAEs) on model activations
- Mechanistic interpretability research
- Analysis of internal model representations
- Debugging and model understanding

## Key Features

- ✅ **Zero overhead when disabled**: No performance impact when `capture_activations=False`
- ✅ **Selective capture**: Choose specific layers and activation types
- ✅ **FSDP compatible**: Works with Fully Sharded Data Parallel training
- ✅ **JAX native**: Compatible with jit, grad, vmap, and other JAX transformations
- ✅ **Clean separation**: Activations stored in separate pytree collection

## Configuration

Add these parameters to your config file or pass them via command line:

```yaml
# Enable activation capture
capture_activations: true

# Capture from specific layers (empty list = all layers)
# Example: layers 0, 6, 12, 18, and 24
activation_capture_layers: [0, 6, 12, 18, 24]

# Types of activations to capture (empty list = all types)
# Options: 'input', 'post_attn', 'pre_mlp', 'post_mlp', 'output'
activation_capture_types: ['post_mlp', 'post_attn', 'output']
```

### Activation Types

| Type | Description | Shape |
|------|-------------|-------|
| `input` | Layer input before any processing | `[batch, seq_len, hidden_dim]` |
| `post_attn` | Attention output before residual addition | `[batch, seq_len, hidden_dim]` |
| `pre_mlp` | Input to MLP block (after post-attn norm) | `[batch, seq_len, hidden_dim]` |
| `post_mlp` | MLP output before residual addition | `[batch, seq_len, hidden_dim]` |
| `output` | Final layer output after all processing | `[batch, seq_len, hidden_dim]` |

## Basic Usage

### Capturing Activations During Forward Pass

```python
import jax
import jax.numpy as jnp
from MaxText import pyconfig

# Load config with activation capture enabled
config = pyconfig.load_config('configs/models/qwen3-0.6b.yml')
config.capture_activations = True
config.activation_capture_layers = [0, 6, 12]
config.activation_capture_types = ['post_mlp']

# Initialize model
model = ...  # Your model initialization

# Forward pass with activation capture
variables = model.apply(
    {'params': params},
    decoder_input_tokens=inputs,
    decoder_positions=positions,
    capture_intermediates=lambda m, method: method == "__call__",
    mutable=['activations']
)

# Extract activations
activations = variables['intermediates']['activations']
# For non-scanned layers: {'post_mlp_layer_0': array(...), 'post_mlp_layer_6': array(...), ...}
# For scanned layers: {'post_mlp': array([num_layers, batch, seq_len, hidden_dim])}
```

### Scanned vs Non-Scanned Layers

The activation format differs based on whether `scan_layers=True`:

**Non-scanned layers** (`scan_layers=False`):
```python
# Activations have layer-specific keys
activations = {
    'post_mlp_layer_0': array([batch, seq_len, hidden_dim]),
    'post_mlp_layer_6': array([batch, seq_len, hidden_dim]),
    'post_mlp_layer_12': array([batch, seq_len, hidden_dim]),
}
```

**Scanned layers** (`scan_layers=True`):
```python
# Activations are stacked along first dimension
activations = {
    'post_mlp': array([num_layers, batch, seq_len, hidden_dim]),
    # Index into first dimension to get specific layer
}

# Get activations from layer 6:
layer_6_activations = activations['post_mlp'][6]
```

## Memory-Efficient Sampling

For large-scale training, capturing activations at every step is memory-intensive. Use periodic sampling:

```python
def training_step_with_periodic_capture(state, batch, step, capture_every=100):
    """Training step with periodic activation capture."""
    
    # Only capture activations every N steps
    should_capture = (step % capture_every == 0)
    
    if should_capture:
        # Forward pass with activation capture
        (logits, hidden_states), variables = state.apply_fn(
            {'params': state.params},
            decoder_input_tokens=batch['inputs'],
            decoder_positions=batch['positions'],
            capture_intermediates=lambda m, method: method == "__call__",
            mutable=['activations']
        )
        
        # Extract and save/process activations
        activations = variables['intermediates']['activations']
        # Save to disk or process for SAE training
        save_activations(activations, step)
    else:
        # Regular forward pass without capture
        logits, hidden_states = state.apply_fn(
            {'params': state.params},
            decoder_input_tokens=batch['inputs'],
            decoder_positions=batch['positions'],
        )
    
    # Compute loss and update
    loss = compute_loss(logits, batch['targets'])
    return loss
```

## SAE Training Pipeline Example

Here's a complete example of collecting activations for SAE training:

```python
import jax
import jax.numpy as jnp
from typing import Dict, List
import numpy as np

class ActivationCollector:
    """Collects activations for SAE training."""
    
    def __init__(self, max_samples: int = 100000):
        self.max_samples = max_samples
        self.collected_activations = []
        
    def collect(self, activations: Dict[str, jax.Array], layer_name: str):
        """Collect activations from a specific layer."""
        if layer_name in activations:
            acts = activations[layer_name]
            # Reshape to [num_tokens, hidden_dim]
            acts_flat = acts.reshape(-1, acts.shape[-1])
            self.collected_activations.append(np.array(acts_flat))
            
            # Trim if we've collected enough
            if sum(a.shape[0] for a in self.collected_activations) > self.max_samples:
                self._trim_to_max()
    
    def _trim_to_max(self):
        """Trim collected activations to max_samples."""
        total = sum(a.shape[0] for a in self.collected_activations)
        if total > self.max_samples:
            all_acts = np.concatenate(self.collected_activations, axis=0)
            self.collected_activations = [all_acts[:self.max_samples]]
    
    def get_dataset(self):
        """Get collected activations as a single array."""
        if not self.collected_activations:
            return None
        return np.concatenate(self.collected_activations, axis=0)

# Usage in training loop
collector = ActivationCollector(max_samples=100000)

for step, batch in enumerate(train_loader):
    if step % 10 == 0:  # Capture every 10 steps
        _, variables = model.apply(
            {'params': params},
            decoder_input_tokens=batch['inputs'],
            decoder_positions=batch['positions'],
            capture_intermediates=lambda m, method: method == "__call__",
            mutable=['activations']
        )
        
        activations = variables['intermediates']['activations']
        collector.collect(activations, 'post_mlp_layer_12')
    
    # ... rest of training step

# Get collected activations for SAE training
sae_training_data = collector.get_dataset()
print(f"Collected {sae_training_data.shape[0]} activation samples")
```

## Distributed/Sharded Activations

When using FSDP or model parallelism, activations are sharded across devices. To gather them:

```python
from jax.experimental import multihost_utils

# Activations are sharded across devices
sharded_activations = variables['intermediates']['activations']

# Gather all shards to get full activations
def gather_activations(sharded_acts):
    """Gather sharded activations from all devices."""
    gathered = {}
    for key, value in sharded_acts.items():
        # Use multihost_utils to gather across hosts
        gathered[key] = multihost_utils.process_allgather(value)
    return gathered

full_activations = gather_activations(sharded_activations)
```

## Performance Considerations

### Compute Overhead
- **Disabled**: < 0.1% overhead (config check only)
- **Enabled**: 1-2% overhead for selective capture
- **Full capture**: 5-10% overhead when capturing all layers and types

### Memory Overhead
- Depends on number of layers and types captured
- Selective capture (5 layers, 1 type): ~5-10% memory increase
- Full capture (all layers, all types): Can double memory usage

### Optimization Tips

1. **Selective Capture**: Only capture the layers and types you need
   ```python
   config.activation_capture_layers = [12]  # Only middle layer
   config.activation_capture_types = ['post_mlp']  # Only MLP output
   ```

2. **Periodic Sampling**: Don't capture every step
   ```python
   # Capture every 100 steps instead of every step
   should_capture = (step % 100 == 0)
   ```

3. **Batch Size**: Consider reducing batch size when capturing
   ```python
   # Use smaller batch for capture steps
   capture_batch_size = batch_size // 2
   ```

4. **Offload to CPU**: Save activations to CPU/disk immediately
   ```python
   import jax
   
   # Move to CPU and save
   cpu_acts = jax.device_put(activations['post_mlp'], jax.devices('cpu')[0])
   np.save('activations.npy', np.array(cpu_acts))
   ```

## Troubleshooting

### Issue: Activations not appearing in output

**Solution**: Make sure to pass `mutable=['activations']` to `model.apply()`:
```python
variables = model.apply(
    {'params': params},
    ...,
    mutable=['activations']  # Required!
)
```

### Issue: Out of memory errors

**Solutions**:
1. Reduce number of captured layers
2. Capture fewer activation types
3. Use periodic sampling (capture every N steps)
4. Reduce batch size during capture steps
5. Save activations to disk immediately

### Issue: Activations have unexpected shape

For scanned layers, activations have shape `[num_layers, batch, seq_len, hidden_dim]`. Index the first dimension:
```python
# Get layer 12 activations
layer_12 = activations['post_mlp'][12]
```

### Issue: Performance degradation

If you're seeing > 2% overhead:
1. Verify `capture_activations=False` when not capturing
2. Use more selective capture (fewer layers/types)
3. Check if you're accidentally capturing every step

## Advanced: Custom Activation Points

To add custom activation capture points, modify the decoder layer:

```python
# In your custom decoder layer
def __call__(self, inputs, ...):
    x = self.some_operation(inputs)
    
    # Add custom capture point
    if self._should_capture('custom_point'):
        self.sow('activations', 'custom_point', x)
    
    return x
```

## References

- [Flax sow/collect documentation](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/variable.html)
- [Sparse Autoencoders for Interpretability](https://transformer-circuits.pub/2024/april-update/index.html)
- [MaxText Documentation](../README.md)

## Support

For issues or questions:
- File an issue on [GitHub](https://github.com/AI-Hypercomputer/maxtext/issues)
- Check existing examples in `examples/extract_activations_qwen3.py`
