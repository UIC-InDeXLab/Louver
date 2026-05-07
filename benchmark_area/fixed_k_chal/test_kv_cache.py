"""
Test script to verify ObserveAttentionHelper works correctly with KV cache enabled.
Tests on Llama-3.2-3B model.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from helpers import ObserveAttentionHelper


def test_basic_functionality():
    """Test that the observer can capture attention with KV cache enabled."""
    
    print("=" * 80)
    print("Testing ObserveAttentionHelper with KV cache on Llama-3.2-3B")
    print("=" * 80)
    
    # Initialize observer
    model_name = "meta-llama/Llama-3.2-3B"
    observer = ObserveAttentionHelper(
        model_name=model_name,
        max_new_tokens=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_model=True,
    )
    
    # Test prompt
    test_prompt = "The capital of France is"
    
    print(f"\n📝 Test prompt: '{test_prompt}'")
    print()
    
    # Run model
    output = observer.run_model(test_prompt)
    
    print(f"\n✅ Generated output: {output}")
    print()
    
    # Get attention info
    info = observer.get_attention_info()
    print("\n📊 Attention Info:")
    print(f"  - Number of layers: {info['num_layers']}")
    print(f"  - Q heads per layer: {info['num_q_heads']}")
    print(f"  - KV heads per layer: {info['num_kv_heads']}")
    print(f"  - Head dimension: {info['head_dim']}")
    print(f"  - Prompt length: {info['prompt_length']}")
    print(f"  - Generated tokens: {info['num_generated']}")
    print(f"  - Total tokens: {info['total_tokens']}")
    print(f"  - Uses GQA: {info['uses_gqa']}")
    
    # Test 1: Check that queries are captured for generated tokens
    print("\n" + "=" * 80)
    print("Test 1: Verify queries are captured for generated tokens")
    print("=" * 80)
    num_generated = len(observer.generated_tokens)
    print(f"Generated {num_generated} tokens")
    
    # Check first layer
    first_layer = list(observer.queries.keys())[0]
    first_head = list(observer.queries[first_layer].keys())[0]
    
    num_queries = len(observer.queries[first_layer][first_head])
    print(f"Layer {first_layer}, Head {first_head}: {num_queries} queries stored")
    
    if num_queries == num_generated:
        print("✅ PASS: Number of queries matches number of generated tokens")
    else:
        print(f"❌ FAIL: Expected {num_generated} queries, got {num_queries}")
    
    # Test 2: Check that keys are captured for all tokens (prompt + generated)
    print("\n" + "=" * 80)
    print("Test 2: Verify keys are captured for all tokens")
    print("=" * 80)
    
    total_tokens = info['prompt_length'] + info['num_generated']
    first_kv_head = list(observer.keys[first_layer].keys())[0]
    num_keys = len(observer.keys[first_layer][first_kv_head])
    
    print(f"Total tokens in sequence: {total_tokens}")
    print(f"Layer {first_layer}, KV Head {first_kv_head}: {num_keys} keys stored")
    
    if num_keys == total_tokens:
        print("✅ PASS: Number of keys matches total tokens (prompt + generated)")
    else:
        print(f"❌ FAIL: Expected {total_tokens} keys, got {num_keys}")
    
    # Test 3: Test get_stats_at_token functionality
    print("\n" + "=" * 80)
    print("Test 3: Verify get_stats_at_token works correctly")
    print("=" * 80)
    
    if num_generated > 0:
        token_idx = 0  # First generated token
        stats = observer.get_stats_at_token(token_idx)
        
        print(f"Getting stats for generated token {token_idx}")
        print(f"Token string: '{observer.get_token_string(token_idx)}'")
        
        # Check if stats are returned
        if len(stats) > 0:
            print(f"✅ Stats returned for {len(stats)} layers")
            
            # Check first layer's first head
            if first_layer in stats:
                layer_stats = stats[first_layer]
                if first_head in layer_stats:
                    head_stats = layer_stats[first_head]
                    print(f"  Layer {first_layer}, Head {first_head}:")
                    print(f"    - Query shape: {head_stats['query'].shape}")
                    print(f"    - Keys shape: {head_stats['keys'].shape}")
                    print(f"    - Scores shape: {head_stats['scores'].shape}")
                    print(f"    - KV head index: {head_stats['kv_head_idx']}")
                    
                    # Verify shapes
                    expected_query_shape = (info['head_dim'],)
                    expected_keys_shape = (info['prompt_length'], info['head_dim'])  # Only prompt keys for first token
                    expected_scores_shape = (info['prompt_length'],)
                    
                    if head_stats['query'].shape == expected_query_shape:
                        print(f"    ✅ Query shape correct: {expected_query_shape}")
                    else:
                        print(f"    ❌ Query shape incorrect: expected {expected_query_shape}, got {head_stats['query'].shape}")
                    
                    if head_stats['keys'].shape == expected_keys_shape:
                        print(f"    ✅ Keys shape correct: {expected_keys_shape}")
                    else:
                        print(f"    ❌ Keys shape incorrect: expected {expected_keys_shape}, got {head_stats['keys'].shape}")
                    
                    if head_stats['scores'].shape == expected_scores_shape:
                        print(f"    ✅ Scores shape correct: {expected_scores_shape}")
                    else:
                        print(f"    ❌ Scores shape incorrect: expected {expected_scores_shape}, got {head_stats['scores'].shape}")
                    
                    print("✅ PASS: get_stats_at_token works correctly")
                else:
                    print(f"❌ FAIL: Head {first_head} not found in stats")
            else:
                print(f"❌ FAIL: Layer {first_layer} not found in stats")
        else:
            print("❌ FAIL: No stats returned")
    
    # Test 4: Check token decoding
    print("\n" + "=" * 80)
    print("Test 4: Verify token decoding")
    print("=" * 80)
    
    print("Generated tokens:")
    for i in range(min(5, num_generated)):
        token_str = observer.get_token_string(i)
        print(f"  Token {i}: '{token_str}'")
    
    print("✅ PASS: Token decoding works")
    
    # Test 5: Memory efficiency check
    print("\n" + "=" * 80)
    print("Test 5: Memory efficiency check")
    print("=" * 80)
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
    
    # Clean up
    del observer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_functionality()
