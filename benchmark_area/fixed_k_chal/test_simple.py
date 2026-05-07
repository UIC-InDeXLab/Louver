"""
Simple test script to verify ObserveAttentionHelper functionality
"""
import torch
from helpers import ObserveAttentionHelper

def test_basic_initialization(device="cpu"):
    """Test that the model loads correctly"""
    print("=" * 60)
    print("TEST 1: Basic Initialization")
    print("=" * 60)
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Using device: {device}")
    observer = ObserveAttentionHelper(model_name, max_new_tokens=5, device=device)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Device: {observer.device}")
    print(f"✓ Max new tokens: {observer.max_new_tokens}")
    print()
    
    return observer


def test_run_model(observer):
    """Test running the model with a simple prompt"""
    print("=" * 60)
    print("TEST 2: Run Model")
    print("=" * 60)
    
    input_text = "Hello, my name is"
    print(f"Input: '{input_text}'")
    print()
    
    try:
        output = observer.run_model(input_text)
        print(f"\n✓ Generation successful!")
        print(f"Output: '{output}'")
        print(f"✓ Generated {len(observer.generated_tokens)} tokens")
        print(f"✓ Prompt length: {observer.prompt_length} tokens")
        print()
        return True
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage(observer):
    """Test that queries and keys are stored correctly"""
    print("=" * 60)
    print("TEST 3: Check Storage & Attention Info")
    print("=" * 60)
    
    # Get comprehensive info
    info = observer.get_attention_info()
    
    print("Attention Information:")
    print(f"✓ Number of layers: {info['num_layers']}")
    print(f"✓ Query heads per layer: {info['num_q_heads']}")
    print(f"✓ Key/Value heads per layer: {info['num_kv_heads']}")
    print(f"✓ Head dimension: {info['head_dim']}")
    print(f"✓ Uses Grouped Query Attention (GQA): {info['uses_gqa']}")
    if info['uses_gqa']:
        head_ratio = info['num_q_heads'] // info['num_kv_heads']
        print(f"  → Each K/V head is shared by {head_ratio} Q heads")
    print()
    
    print("Token Information:")
    print(f"✓ Prompt length: {info['prompt_length']} tokens")
    print(f"✓ Generated tokens: {info['num_generated']} tokens")
    print(f"✓ Total tokens: {info['total_tokens']} tokens")
    print()
    
    print("Tokens available for attention at each generation step:")
    for pos_info in info['tokens_at_position']:
        print(f"  Token {pos_info['token_index']}: {pos_info['tokens_available']} tokens available "
              f"({pos_info['prompt_tokens']} prompt + {pos_info['prev_generated']} prev generated)")
    print()
    
    print(f"Number of layers: {len(observer.queries)}")
    
    if len(observer.queries) > 0:
        layer_0 = list(observer.queries.keys())[0]
        print(f"✓ Layer 0 has {len(observer.queries[layer_0])} heads")
        
        head_0 = list(observer.queries[layer_0].keys())[0]
        print(f"✓ Head 0 has queries for {len(observer.queries[layer_0][head_0])} tokens")
        
        if len(observer.queries[layer_0][head_0]) > 0:
            token_0 = list(observer.queries[layer_0][head_0].keys())[0]
            q_shape = observer.queries[layer_0][head_0][token_0].shape
            print(f"✓ Query shape: {q_shape}")
        
        # Check keys
        print(f"✓ Head 0 has keys for {len(observer.keys[layer_0][head_0])} tokens")
        if len(observer.keys[layer_0][head_0]) > 0:
            key_0 = list(observer.keys[layer_0][head_0].keys())[0]
            k_shape = observer.keys[layer_0][head_0][key_0].shape
            print(f"✓ Key shape: {k_shape}")
        print()
        return True
    else:
        print("✗ No data captured!")
        return False


def test_get_stats(observer):
    """Test getting stats for a specific token"""
    print("=" * 60)
    print("TEST 4: Get Stats at Token")
    print("=" * 60)
    
    if len(observer.generated_tokens) == 0:
        print("✗ No generated tokens to test")
        return False
    
    try:
        # Get stats for first generated token
        token_idx = 0
        stats = observer.get_stats_at_token(token_idx)
        
        # Get the actual token string
        token_string = observer.get_token_string(token_idx)
        
        print(f"Stats for generated token {token_idx}: '{token_string}'")
        print(f"✓ Number of layers: {len(stats)}")
        
        if len(stats) > 0:
            layer_0 = list(stats.keys())[0]
            print(f"✓ Layer 0 has {len(stats[layer_0])} query heads")
            
            if len(stats[layer_0]) > 0:
                head_0 = list(stats[layer_0].keys())[0]
                print(f"✓ Query head 0 data:")
                print(f"  - Query shape: {stats[layer_0][head_0]['query'].shape}")
                print(f"  - Keys shape: {stats[layer_0][head_0]['keys'].shape}")
                print(f"  - Scores shape: {stats[layer_0][head_0]['scores'].shape}")
                print(f"  - Number of keys (should be prompt_length): {stats[layer_0][head_0]['keys'].shape[0]}")
                if 'kv_head_idx' in stats[layer_0][head_0]:
                    print(f"  - K/V head index (for GQA): {stats[layer_0][head_0]['kv_head_idx']}")
        
        # Test a few more tokens
        print(f"\nGenerated tokens:")
        for i in range(min(5, len(observer.generated_tokens))):
            token_str = observer.get_token_string(i)
            print(f"  Token {i}: '{token_str}'")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Error getting stats: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("ObserveAttentionHelper Test Suite")
    print("="*60 + "\n")
    
    # Allow device selection from command line
    import sys
    device = "cpu"  # Default to CPU to avoid memory issues
    if len(sys.argv) > 1:
        device = sys.argv[1]
    
    print(f"NOTE: Running on {device.upper()}")
    if device == "cpu":
        print("NOTE: This will be slow but avoids GPU memory issues")
    print()
    
    try:
        observer = test_basic_initialization(device=device)
        
        success = test_run_model(observer)
        if not success:
            print("\n✗ Tests stopped due to model run failure")
            return
        
        test_storage(observer)
        test_get_stats(observer)
        
        print("=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
