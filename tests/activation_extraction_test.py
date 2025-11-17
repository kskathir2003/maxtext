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

"""Tests for activation extraction in Qwen3 layers."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import nnx

from MaxText import pyconfig
from MaxText.layers import qwen3
from MaxText.layers import nnx_wrappers
from MaxText.globals import MAXTEXT_PKG_DIR


class ActivationExtractionTest(unittest.TestCase):
  """Tests for activation extraction feature."""

  def setUp(self):
    """Set up test configuration."""
    self.config = pyconfig.load_config(f"{MAXTEXT_PKG_DIR}/configs/base.yml")
    # Override with minimal config for testing
    self.config.decoder_block = 'qwen3'
    self.config.emb_dim = 128
    self.config.mlp_dim = 256
    self.config.num_query_heads = 4
    self.config.num_kv_heads = 4
    self.config.head_dim = 32
    self.config.vocab_size = 1000
    self.config.max_target_length = 128
    self.config.max_prefill_predict_length = 128
    self.config.dtype = 'float32'
    self.config.weight_dtype = 'float32'
    self.config.dropout_rate = 0.0
    self.config.normalization_layer_epsilon = 1e-5
    self.config.mlp_activations = ['silu', 'linear']
    self.config.attention = 'dot_product'
    self.config.float32_qk_product = False
    self.config.float32_logits = False
    self.config.use_qk_norm = False
    
  def test_capture_disabled_by_default(self):
    """Test that activation capture is disabled by default."""
    self.assertFalse(self.config.capture_activations)
    
  def test_qwen3_decoder_layer_has_layer_index(self):
    """Test that Qwen3DecoderLayer accepts and stores layer_index."""
    mesh = Mesh(jax.devices(), ('data',))
    rngs = nnx.Rngs(0)
    
    layer = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=5
    )
    
    self.assertEqual(layer.layer_index, 5)
    
  def test_should_capture_logic(self):
    """Test _should_capture method logic."""
    mesh = Mesh(jax.devices(), ('data',))
    rngs = nnx.Rngs(0)
    
    # Test 1: Capture disabled
    self.config.capture_activations = False
    layer = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=0
    )
    self.assertFalse(layer._should_capture('post_mlp'))
    
    # Test 2: Capture enabled, all layers, all types
    self.config.capture_activations = True
    self.config.activation_capture_layers = []
    self.config.activation_capture_types = []
    layer = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=5
    )
    self.assertTrue(layer._should_capture('post_mlp'))
    self.assertTrue(layer._should_capture('input'))
    
    # Test 3: Capture enabled, specific layers only
    self.config.activation_capture_layers = [0, 5, 10]
    layer_match = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=5
    )
    self.assertTrue(layer_match._should_capture('post_mlp'))
    
    layer_no_match = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=3
    )
    self.assertFalse(layer_no_match._should_capture('post_mlp'))
    
    # Test 4: Capture enabled, specific types only
    self.config.activation_capture_layers = []
    self.config.activation_capture_types = ['post_mlp', 'output']
    layer = qwen3.Qwen3DecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=5
    )
    self.assertTrue(layer._should_capture('post_mlp'))
    self.assertTrue(layer._should_capture('output'))
    self.assertFalse(layer._should_capture('input'))
    self.assertFalse(layer._should_capture('pre_mlp'))
    
  def test_activation_capture_non_scanned(self):
    """Test activation capture with non-scanned layers."""
    # Enable activation capture
    self.config.capture_activations = True
    self.config.activation_capture_layers = [0]
    self.config.activation_capture_types = ['post_mlp', 'output']
    self.config.scan_layers = False
    
    mesh = Mesh(jax.devices(), ('data',))
    
    # Create layer using Linen wrapper
    LinenLayer = nnx_wrappers.to_linen_class(
        qwen3.Qwen3DecoderLayer,
        base_metadata_fn=lambda x: None
    )
    
    # Create dummy inputs
    batch_size, seq_len = 2, 16
    inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    # Initialize and apply layer
    variables = LinenLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        layer_index=0
    ).init(
        jax.random.PRNGKey(0),
        inputs,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode='train',
    )
    
    # Apply with activation capture
    output, mutated = LinenLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        layer_index=0
    ).apply(
        variables,
        inputs,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode='train',
        mutable=['activations']
    )
    
    # Check that activations were captured
    self.assertIn('activations', mutated)
    activations = mutated['activations']
    
    # Check expected keys exist
    self.assertIn('post_mlp_layer_0', activations)
    self.assertIn('output_layer_0', activations)
    
    # Check shapes are correct
    expected_shape = (batch_size, seq_len, self.config.emb_dim)
    self.assertEqual(activations['post_mlp_layer_0'].shape, expected_shape)
    self.assertEqual(activations['output_layer_0'].shape, expected_shape)
    
    # Check that non-captured types are not present
    self.assertNotIn('input_layer_0', activations)
    self.assertNotIn('pre_mlp_layer_0', activations)
    
  def test_moe_layer_has_layer_index(self):
    """Test that Qwen3MoeDecoderLayer accepts and stores layer_index."""
    self.config.num_experts = 8
    self.config.num_experts_per_tok = 2
    self.config.moe_mlp_dim = 256
    
    mesh = Mesh(jax.devices(), ('data',))
    rngs = nnx.Rngs(0)
    
    layer = qwen3.Qwen3MoeDecoderLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        rngs=rngs,
        layer_index=7
    )
    
    self.assertEqual(layer.layer_index, 7)
    
  def test_no_overhead_when_disabled(self):
    """Test that there's minimal overhead when capture is disabled."""
    self.config.capture_activations = False
    self.config.scan_layers = False
    
    mesh = Mesh(jax.devices(), ('data',))
    
    LinenLayer = nnx_wrappers.to_linen_class(
        qwen3.Qwen3DecoderLayer,
        base_metadata_fn=lambda x: None
    )
    
    batch_size, seq_len = 2, 16
    inputs = jnp.ones((batch_size, seq_len, self.config.emb_dim))
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    variables = LinenLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        layer_index=0
    ).init(
        jax.random.PRNGKey(0),
        inputs,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode='train',
    )
    
    # Apply without mutable=['activations']
    output = LinenLayer(
        config=self.config,
        mesh=mesh,
        model_mode='train',
        quant=None,
        layer_index=0
    ).apply(
        variables,
        inputs,
        decoder_segment_ids=None,
        decoder_positions=positions,
        deterministic=True,
        model_mode='train',
    )
    
    # Should complete without errors
    self.assertEqual(output.shape, (batch_size, seq_len, self.config.emb_dim))


if __name__ == '__main__':
  unittest.main()
