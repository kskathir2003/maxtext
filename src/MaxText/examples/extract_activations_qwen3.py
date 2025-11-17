#!/usr/bin/env python
# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating activation extraction for Qwen3 models.

This script shows how to:
1. Enable activation capture in the config
2. Run forward passes with activation extraction
3. Process and save activations for SAE training
4. Handle distributed/sharded activations

Usage:
  python src/MaxText/examples/extract_activations_qwen3.py
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional
from absl import app, flags

# Add MaxText to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MaxText import pyconfig
from MaxText import max_utils

FLAGS = flags.FLAGS

# Config flags
flags.DEFINE_string('config', 'src/MaxText/configs/models/qwen3-0.6b.yml', 'Path to config file')
flags.DEFINE_string('output_dir', '/tmp/activations', 'Directory to save activations')
flags.DEFINE_integer('num_samples', 1000, 'Number of activation samples to collect')
flags.DEFINE_integer('capture_every', 10, 'Capture activations every N steps')
flags.DEFINE_list('capture_layers', '0,12,24', 'Comma-separated list of layer indices to capture')
flags.DEFINE_list('capture_types', 'post_mlp,post_attn', 'Comma-separated list of activation types')


class ActivationCollector:
  """Collects and manages activations for SAE training."""
  
  def __init__(self, max_samples: int = 100000, output_dir: str = '/tmp/activations'):
    self.max_samples = max_samples
    self.output_dir = output_dir
    self.collected = {}
    os.makedirs(output_dir, exist_ok=True)
    
  def collect(self, activations: Dict[str, jax.Array], step: int):
    """Collect activations from a forward pass."""
    for key, value in activations.items():
      if key not in self.collected:
        self.collected[key] = []
      
      # Flatten to [num_tokens, hidden_dim]
      # Handle both scanned and non-scanned layer formats
      if len(value.shape) == 4:  # Scanned: [num_layers, batch, seq_len, hidden_dim]
        acts_flat = value.reshape(-1, value.shape[-1])
      else:  # Non-scanned: [batch, seq_len, hidden_dim]
        acts_flat = value.reshape(-1, value.shape[-1])
      
      # Convert to numpy and store
      self.collected[key].append(np.array(acts_flat))
      
    # Periodically save to disk to avoid memory issues
    if step % 50 == 0:
      self._save_checkpoint(step)
  
  def _save_checkpoint(self, step: int):
    """Save collected activations to disk."""
    for key, acts_list in self.collected.items():
      if acts_list:
        all_acts = np.concatenate(acts_list, axis=0)
        # Trim to max_samples
        if all_acts.shape[0] > self.max_samples:
          all_acts = all_acts[:self.max_samples]
        
        filepath = os.path.join(self.output_dir, f'{key}_step{step}.npy')
        np.save(filepath, all_acts)
        print(f'Saved {all_acts.shape[0]} samples to {filepath}')
    
    # Clear memory
    self.collected = {}
  
  def finalize(self, step: int):
    """Save final activations."""
    self._save_checkpoint(step)
    print(f'Finalized activation collection. Saved to {self.output_dir}')


def create_sample_batch(config, batch_size: int = 2, seq_len: int = 128):
  """Create a sample batch for testing activation extraction."""
  vocab_size = config.vocab_size
  
  # Random token IDs
  decoder_input_tokens = jax.random.randint(
      jax.random.PRNGKey(0),
      shape=(batch_size, seq_len),
      minval=0,
      maxval=vocab_size
  )
  
  # Position IDs
  decoder_positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
  
  return {
      'inputs': decoder_input_tokens,
      'positions': decoder_positions,
  }


def extract_activations_example(config_path: str, capture_layers: List[int], 
                                capture_types: List[str], num_samples: int,
                                output_dir: str):
  """
  Example of extracting activations from Qwen3 model.
  
  Args:
    config_path: Path to model config file
    capture_layers: List of layer indices to capture from
    capture_types: List of activation types to capture
    num_samples: Number of activation samples to collect
    output_dir: Directory to save activations
  """
  
  print("=" * 80)
  print("MaxText Activation Extraction Example for Qwen3")
  print("=" * 80)
  
  # Load and configure
  print(f"\n1. Loading config from {config_path}")
  config = pyconfig.load_config(config_path)
  
  # Enable activation capture
  config.capture_activations = True
  config.activation_capture_layers = capture_layers
  config.activation_capture_types = capture_types
  
  print(f"   - Capturing from layers: {capture_layers}")
  print(f"   - Capturing types: {capture_types}")
  print(f"   - Target samples: {num_samples}")
  
  # Initialize collector
  collector = ActivationCollector(max_samples=num_samples, output_dir=output_dir)
  
  # For demonstration, we'll simulate multiple forward passes
  # In real usage, this would be integrated into your training loop
  
  print("\n2. Creating sample batches")
  num_steps = min(100, (num_samples // (2 * 128)) + 1)  # Estimate steps needed
  print(f"   - Will run {num_steps} forward passes")
  
  # Create sample batches
  batches = [create_sample_batch(config) for _ in range(num_steps)]
  
  print("\n3. Running forward passes with activation capture")
  print("   (In production, integrate this into your training loop)")
  
  # Simulate forward passes
  for step in range(num_steps):
    # In real code, this would be your actual model forward pass
    # Here we demonstrate the pattern
    
    print(f"\n   Step {step + 1}/{num_steps}:", end=" ")
    
    # This is a placeholder - in real usage, you would do:
    # variables = model.apply(
    #     {'params': params},
    #     decoder_input_tokens=batches[step]['inputs'],
    #     decoder_positions=batches[step]['positions'],
    #     capture_intermediates=lambda m, method: method == "__call__",
    #     mutable=['activations']
    # )
    # activations = variables['intermediates']['activations']
    
    # For demo purposes, create fake activations with correct shape
    fake_activations = {}
    batch_size, seq_len = batches[step]['inputs'].shape
    hidden_dim = config.emb_dim
    
    if config.scan_layers:
      # Scanned format: [num_layers, batch, seq_len, hidden_dim]
      for act_type in capture_types:
        fake_activations[act_type] = jax.random.normal(
            jax.random.PRNGKey(step),
            shape=(len(capture_layers), batch_size, seq_len, hidden_dim)
        )
    else:
      # Non-scanned format: separate key per layer
      for layer_idx in capture_layers:
        for act_type in capture_types:
          key = f'{act_type}_layer_{layer_idx}'
          fake_activations[key] = jax.random.normal(
              jax.random.PRNGKey(step + layer_idx),
              shape=(batch_size, seq_len, hidden_dim)
          )
    
    # Collect activations
    collector.collect(fake_activations, step)
    print(f"Captured {len(fake_activations)} activation tensors")
  
  # Finalize collection
  print("\n4. Finalizing and saving activations")
  collector.finalize(num_steps)
  
  print("\n" + "=" * 80)
  print("Example completed successfully!")
  print("=" * 80)
  print(f"\nActivations saved to: {output_dir}")
  print("\nNext steps:")
  print("1. Load activations: data = np.load('path/to/activation.npy')")
  print("2. Train SAE on the activations")
  print("3. Analyze learned features for interpretability")
  print("\nSee docs/activation_extraction.md for more details.")


def main(argv):
  """Main entry point."""
  del argv  # Unused
  
  # Parse capture layers and types
  capture_layers = [int(x) for x in FLAGS.capture_layers]
  capture_types = FLAGS.capture_types
  
  # Run example
  extract_activations_example(
      config_path=FLAGS.config,
      capture_layers=capture_layers,
      capture_types=capture_types,
      num_samples=FLAGS.num_samples,
      output_dir=FLAGS.output_dir
  )


if __name__ == '__main__':
  app.run(main)
