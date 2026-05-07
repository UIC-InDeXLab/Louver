import torch
import pickle
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class ObserveAttentionHelper:
    def __init__(
        self,
        model_name,
        max_new_tokens=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_model=True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model = None
        self.tokenizer = None

        # Storage for queries and keys
        # Structure: {layer_idx: {head_idx: {token_idx: tensor}}}
        self.queries = {}  # Queries for generated tokens
        self.keys = {}  # Keys for all tokens (prompt + generated)
        self.keys_tensor = None  # Cached tensor version of keys
        self.queries_tensor = None  # Cached tensor version of queries
        self.prompt_length = 0
        self.generated_tokens = []
        self.all_token_ids = []

        if load_model:
            self._prepare_model()

    def _prepare_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            output_attentions=False,  # We'll capture manually via hooks
        )
        self.model.eval()

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.device}")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        module,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(module, "rotary_emb"):
            return q, k

        if position_ids is None:
            if q.shape[1] == 1 and self.current_generation_step > 0:
                pos = self.prompt_length + self.current_generation_step - 1
                position_ids = torch.tensor(
                    [[pos]], device=hidden_states.device, dtype=torch.long
                )
            else:
                position_ids = torch.arange(
                    q.shape[1], device=hidden_states.device, dtype=torch.long
                ).unsqueeze(0)

        cos, sin = module.rotary_emb(k, position_ids)

        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.dim() == 4 and cos.shape[1] == 1:
            cos = cos.permute(0, 2, 1, 3)
            sin = sin.permute(0, 2, 1, 3)
        if cos.dim() == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        return q, k

    def run_model(self, input_text):
        """
        1. Run the model on the input.
        2. For each token being generated during the decoding phase,
            2.1. store the query vector for that token
            2.2. store all the key values for all the tokens generated so far
        """
        # Reset storage
        self.queries = {}
        self.keys = {}
        self.prompt_length = 0
        self.generated_tokens = []
        self.all_token_ids = []  # Store full sequence: prompt + generated

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        self.prompt_length = inputs.input_ids.shape[1]

        print(f"Prompt length: {self.prompt_length} tokens")
        print(f"Generating {self.max_new_tokens} tokens...")

        # Register hooks to capture Q and K
        hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_pre_hook(
                self._make_attention_hook(layer_idx), with_kwargs=True
            )
            hooks.append(hook)

        # Track current generation step for hooks
        self.current_generation_step = -1  # -1 means prompt processing

        try:
            with torch.no_grad():
                # Generate tokens one at a time with KV cache
                generated_ids = inputs.input_ids.clone()
                past_key_values = None

                for step in tqdm(
                    range(self.max_new_tokens + 1)
                ):  # +1 to capture last token's Q/K
                    # Set generation step before forward pass (so hook can access it)
                    self.current_generation_step = step

                    # For first step, use full sequence; for subsequent steps, only the last token
                    input_ids = generated_ids if step == 0 else generated_ids[:, -1:]

                    outputs = self.model(
                        input_ids=input_ids,
                        use_cache=True,  # Enable KV cache for efficiency
                        past_key_values=past_key_values,
                    )

                    # Update cache for next iteration
                    past_key_values = outputs.past_key_values

                    # Skip token generation on the last extra step (only for observation)
                    if step >= self.max_new_tokens:
                        break

                    # Get next token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Store generated token
                    self.generated_tokens.append(next_token.item())

                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    # Stop if EOS token (but still try to do one more forward pass for observation)
                    # if next_token.item() == self.tokenizer.eos_token_id: # TODO: force all tokens
                    # Do one more forward pass to capture the last token's Q/K
                    # self.current_generation_step = step + 1
                    # final_outputs = self.model(
                    #     input_ids=generated_ids[:, -1:],
                    #     use_cache=True,
                    #     past_key_values=past_key_values,
                    # )
                    # break

                    # if (step + 1) % 10 == 0:
                    # print(f"Generated {step + 1} tokens...", end="\r")

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        print(
            f"\nGeneration complete. Total tokens generated: {len(self.generated_tokens)}"
        )
        print(f"Layers captured: {len(self.queries)}")

        # Store full sequence for later decoding
        self.all_token_ids = generated_ids[0].cpu().tolist()

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _make_attention_hook(self, layer_idx):
        """Create a hook function for a specific layer. Works with KV cache enabled."""

        def hook_fn(module, args, kwargs):
            # Extract hidden_states from kwargs (for newer transformers versions)
            # or from args (for older versions)
            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            elif len(args) > 0:
                hidden_states = args[0]
            else:
                return  # Can't extract hidden states, skip this hook

            # Shape: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Get Q, K projections
            # Note: This is specific to LLaMA-style models
            # For other architectures, this may need adjustment
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)

            position_ids = (
                kwargs.get("position_ids") if isinstance(kwargs, dict) else None
            )

            # Handle Grouped Query Attention (GQA)
            # Q uses num_heads, K/V use num_key_value_heads
            num_q_heads = module.config.num_attention_heads
            num_kv_heads = module.config.num_key_value_heads
            head_dim = module.head_dim

            # Reshape: (batch, seq_len, num_heads, head_dim)
            q = q.view(batch_size, seq_len, num_q_heads, head_dim)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim)

            q, k = self._apply_rope(module, q, k, position_ids, hidden_states)

            # Initialize storage for this layer if needed
            if layer_idx not in self.queries:
                self.queries[layer_idx] = {}
                self.keys[layer_idx] = {}

            if seq_len > 1:
                # PHASE 1: Processing prompt (first forward pass)
                # Store all prompt keys (no queries yet, as we're not generating)
                for head_idx in range(num_kv_heads):
                    if head_idx not in self.keys[layer_idx]:
                        self.keys[layer_idx][head_idx] = {}

                    for pos in range(seq_len):
                        self.keys[layer_idx][head_idx][pos] = (
                            k[0, pos, head_idx, :].detach().cpu().clone()
                        )

            else:  # seq_len == 1
                # PHASE 2: Generating tokens (one at a time with KV cache)
                # We're feeding the token generated at step (current_generation_step - 1)
                # So we capture its query/key under index (current_generation_step - 1)
                if self.current_generation_step > 0:
                    token_idx = self.current_generation_step - 1
                    current_pos = self.prompt_length + token_idx

                    # Store query for the generated token
                    for head_idx in range(num_q_heads):
                        if head_idx not in self.queries[layer_idx]:
                            self.queries[layer_idx][head_idx] = {}

                        self.queries[layer_idx][head_idx][token_idx] = (
                            q[0, 0, head_idx, :].detach().cpu().clone()
                        )

                    # Store key for the generated token
                    for head_idx in range(num_kv_heads):
                        if head_idx not in self.keys[layer_idx]:
                            self.keys[layer_idx][head_idx] = {}

                        self.keys[layer_idx][head_idx][current_pos] = (
                            k[0, 0, head_idx, :].detach().cpu().clone()
                        )

        return hook_fn

    def get_attention_info(self):
        """
        Get comprehensive information about the captured attention data.

        Returns:
            Dict containing:
            - num_layers: Number of transformer layers
            - num_q_heads: Number of query heads per layer
            - num_kv_heads: Number of key/value heads per layer (for GQA)
            - head_dim: Dimension of each attention head
            - prompt_length: Number of tokens in the input prompt
            - num_generated: Number of tokens generated
            - total_tokens: Total tokens (prompt + generated)
            - uses_gqa: Whether the model uses Grouped Query Attention
        """
        info = {
            "num_layers": len(self.queries),
            "num_q_heads": 0,
            "num_kv_heads": 0,
            "head_dim": 0,
            "prompt_length": self.prompt_length,
            "num_generated": len(self.generated_tokens),
            "total_tokens": self.prompt_length + len(self.generated_tokens),
            "uses_gqa": False,
        }

        # Get head information from first layer
        if len(self.queries) > 0:
            first_layer = list(self.queries.keys())[0]
            info["num_q_heads"] = len(self.queries[first_layer])

            if first_layer in self.keys:
                info["num_kv_heads"] = len(self.keys[first_layer])
                info["uses_gqa"] = info["num_q_heads"] != info["num_kv_heads"]

            # Get head dimension from first available query
            if info["num_q_heads"] > 0:
                first_head = list(self.queries[first_layer].keys())[0]
                if len(self.queries[first_layer][first_head]) > 0:
                    first_token = list(self.queries[first_layer][first_head].keys())[0]
                    info["head_dim"] = self.queries[first_layer][first_head][
                        first_token
                    ].shape[0]

        return info

    def get_token_string(self, token_index):
        """
        Get the decoded string representation of a generated token.

        Args:
            token_index: Index of the generated token (0-indexed, 0 = first generated token)

        Returns:
            str: The decoded token string
        """
        if token_index < 0 or token_index >= len(self.generated_tokens):
            raise ValueError(
                f"Invalid token_index {token_index}. Must be in [0, {len(self.generated_tokens)-1}]"
            )

        token_id = self.generated_tokens[token_index]
        token_string = self.tokenizer.decode([token_id])

        return token_string

    def get_token_string_at_position(self, position):
        """
        Get the decoded string of the token at a specific position in the full sequence.

        Args:
            position: Position in the full sequence (0 = first prompt token,
                     prompt_length = first generated token)

        Returns:
            str: The decoded token string
        """
        if position < 0 or position >= len(self.all_token_ids):
            raise ValueError(
                f"Invalid position {position}. Must be in [0, {len(self.all_token_ids)-1}]"
            )

        token_id = self.all_token_ids[position]
        token_string = self.tokenizer.decode([token_id])

        return token_string

    def get_stats_at_token(self, token_index):
        """
        For the i-th generated token, return:
        1. the query vector for that token
        2. the key values for all the tokens generated so far (excluding that token)
        3. scores = query @ keys.T

        Note: For models using Grouped Query Attention (GQA), Q heads are mapped to K/V heads.
        Multiple Q heads may share the same K/V head.

        Args:
            token_index: Index of the generated token (0-indexed, 0 = first generated token)

        Returns:
            Dict with structure:
            {
                layer_idx: {
                    q_head_idx: {
                        'query': tensor of shape (head_dim,),
                        'keys': tensor of shape (num_keys, head_dim),
                        'scores': tensor of shape (num_keys,),
                        'kv_head_idx': index of the K/V head used (for GQA models)
                    }
                }
            }
        """
        if token_index < 0 or token_index >= len(self.generated_tokens):
            raise ValueError(
                f"Invalid token_index {token_index}. Must be in [0, {len(self.generated_tokens)-1}]"
            )

        stats = {}

        # Global position of this token (prompt_length + token_index)
        global_pos = self.prompt_length + token_index

        for layer_idx in self.queries.keys():
            stats[layer_idx] = {}

            # Determine the ratio of Q heads to K/V heads for GQA
            num_q_heads = len(self.queries[layer_idx])
            num_kv_heads = len(self.keys[layer_idx]) if layer_idx in self.keys else 0

            if num_kv_heads == 0:
                continue

            head_ratio = (
                num_q_heads // num_kv_heads
            )  # How many Q heads share one K/V head

            for q_head_idx in self.queries[layer_idx].keys():
                if token_index not in self.queries[layer_idx][q_head_idx]:
                    continue

                query = self.queries[layer_idx][q_head_idx][token_index]

                # Map Q head to corresponding K/V head (for GQA)
                kv_head_idx = q_head_idx // head_ratio

                if kv_head_idx not in self.keys[layer_idx]:
                    continue

                # Collect all keys EXCEPT the current token
                # Include prompt tokens + generated tokens before this one
                keys_list = []
                position_map = []  # Track which position each score corresponds to
                for key_pos in sorted(self.keys[layer_idx][kv_head_idx].keys()):
                    if key_pos < global_pos:  # Exclude current token
                        keys_list.append(self.keys[layer_idx][kv_head_idx][key_pos])
                        position_map.append(key_pos)

                if len(keys_list) > 0:
                    keys = torch.stack(keys_list)  # Shape: (num_keys, head_dim)

                    # Compute scores: query @ keys.T
                    scores = torch.matmul(query, keys.T) / (query.shape[0] ** 0.5)

                    stats[layer_idx][q_head_idx] = {
                        "query": query,
                        "keys": keys,
                        "scores": scores,
                        "kv_head_idx": kv_head_idx,
                        "position_map": position_map,  # Maps score index to sequence position
                    }

        return stats

    def get_keys_tensor(self) -> torch.Tensor:
        if self.keys_tensor is not None:
            return self.keys_tensor

        num_layers = len(self.keys)
        # Infer dimensions from the first populated entry
        first_layer = next(iter(self.keys.values()))
        num_kv_heads = len(first_layer)
        first_head = next(iter(first_layer.values()))
        total_tokens = len(first_head)  # should equal prompt_length + num_generated
        head_dim = next(iter(first_head.values())).shape[0]

        tensor = torch.zeros(num_layers, num_kv_heads, total_tokens, head_dim)

        for layer_idx, heads in tqdm(self.keys.items(), desc="key"):
            for head_idx, positions in heads.items():
                for pos, vec in positions.items():
                    tensor[layer_idx, head_idx, pos] = vec

        return tensor

    def get_queries_tensor(self) -> torch.Tensor:
        if self.queries_tensor is not None:
            return self.queries_tensor

        num_layers = len(self.queries)
        first_layer = next(iter(self.queries.values()))
        num_q_heads = len(first_layer)
        first_head = next(iter(first_layer.values()))
        num_generated = len(first_head)
        head_dim = next(iter(first_head.values())).shape[0]

        tensor = torch.zeros(num_layers, num_q_heads, num_generated, head_dim)

        for layer_idx, heads in tqdm(sorted(self.queries.items()), desc="query"):
            for head_idx, positions in heads.items():
                for token_idx, vec in positions.items():
                    tensor[layer_idx, head_idx, token_idx] = vec

        return tensor

    def snapshot(self):
        keys = self.get_keys_tensor()
        self.keys_tensor = keys
        queries = self.get_queries_tensor()
        self.queries_tensor = queries
        torch.save(keys, "keys.pt")
        torch.save(queries, "queries.pt")

    def save(self, path):
        """
        Save the captured attention data to a file.

        This saves queries, keys, and metadata but NOT the model or tokenizer.

        Args:
            path: Path where the data should be saved (will create parent dirs if needed)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "queries": self.queries,
            "keys": self.keys,
            "prompt_length": self.prompt_length,
            "generated_tokens": self.generated_tokens,
            "all_token_ids": self.all_token_ids,
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
        }

        torch.save(data, path)
        print(f"Data saved to {path}")

    @classmethod
    def from_file(cls, path, device: Optional[str] = None):
        """
        Load captured attention data from a file.

        Creates an ObserveAttentionHelper instance with the saved data
        without loading the model. The tokenizer is loaded for decoding tokens.

        Args:
            path: Path to the saved data file
            device: Device to use (defaults to cuda if available, else cpu)

        Returns:
            ObserveAttentionHelper instance with loaded data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved data found at {path}")

        print(f"Loading data from {path}...")
        data = torch.load(path, map_location="cpu")

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create instance without loading the model
        instance = cls(
            model_name=data["model_name"],
            max_new_tokens=data["max_new_tokens"],
            device=device,
            load_model=False,
        )

        # Load the tokenizer (needed for decoding)
        print(f"Loading tokenizer: {data['model_name']}")
        instance.tokenizer = AutoTokenizer.from_pretrained(data["model_name"])
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        # Restore captured data
        instance.queries = data["queries"]
        instance.keys = data["keys"]
        instance.prompt_length = data["prompt_length"]
        instance.generated_tokens = data["generated_tokens"]
        instance.all_token_ids = data["all_token_ids"]

        print(f"Data loaded successfully")
        print(f"  Layers: {len(instance.queries)}")
        print(f"  Prompt tokens: {instance.prompt_length}")
        print(f"  Generated tokens: {len(instance.generated_tokens)}")

        return instance
