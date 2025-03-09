import math
import torch
import matplotlib.pyplot as plt
from nnsight import CONFIG
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from dataclasses import dataclass, field
import torch
from typing import Optional, Union
import nnsight
from collections import defaultdict
import os
from datetime import datetime
import numpy as np


@dataclass
class ActivationContainer:
    """
    A dataclass to hold activations of keys, values
    at specified layers and token indices
    """

    # mapping from (token idx, layer idx)
    # to another dictionary, mapping string (eg: "key")
    # to tensor
    activity_dict = defaultdict(dict)

    # map the token index to its values
    # eg: 7: ("the ", 311)
    token_idx_to_value_dict = dict()

    # Given a token string, or token int
    # return the corresponding token indices
    # eg: "the " -> returns all index positions of "the "
    token_int_to_token_idx = defaultdict(list)
    token_str_to_token_idx = defaultdict(list)

    def set_tokens(self, token_index, token_int, token_string):
        self.token_idx_to_value_dict[token_index] = (token_string, token_int)

        self.token_int_to_token_idx[token_int].append(token_index)
        self.token_int_to_token_idx[token_string].append(token_index)

        self.token_int_to_token_idx[token_int].sort()
        self.token_int_to_token_idx[token_string].sort()

    def get_token_by_index(self, token_index):
        return self.token_idx_to_value_dict.get(token_index, None)

    def set_activation(self, token_index, layer_index, tensor, label):
        self.activity_dict[(token_index, layer_index)][label] = tensor

    def get_activation(self, token_index, layer_index, label):
        return self.activity_dict[(token_index, layer_index)][label]

    def get_activations_at_token_idx(self, token_index, label):
        output = []
        layer_index = 0
        while (token_index, layer_index) in self.activity_dict:
            output.append(self.activity_dict[(token_index, layer_index)][label])
            layer_index += 1
        return output


@dataclass
class AttentionContainer:
    """
    A container to hold attention maps, accessed by layer.

    Attributes:
        string: Identifier string for this container
        attention_by_layer: Dictionary mapping layer indices to attention tensors
    """

    string: str
    attention_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure string is not empty
        if not self.string:
            raise ValueError("Container identifier string cannot be empty")

    def set_attention(self, layer_idx: int, attention: torch.Tensor):
        """
        Store attention maps for a specific layer.

        Args:
            layer_idx: Index of the layer
            attention: Attention tensor with shape [batch_size, num_heads, seq_len, seq_len]
                       where batch_size should be 1

        Raises:
            ValueError: If batch size is not 1 or tensor shape is invalid
        """
        # Input validation
        if not isinstance(attention, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(attention)}")

        if len(attention.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor [batch, heads, seq_len, seq_len], got shape {attention.shape}"
            )

        if attention.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {attention.shape[0]}")

        # Remove batch dimension and store
        self.attention_by_layer[layer_idx] = attention.squeeze(
            0
        )  # [num_heads, seq_len, seq_len]

    def get_attention(
        self, layer_idx: int, head_idx: Optional[Union[int, list[int]]] = None
    ):
        """
        Retrieve attention maps for a specific layer and optionally specific heads.

        Args:
            layer_idx: Index of the layer
            head_idx: If None, return all heads. If int, return that specific head.
                      If list of ints, return those specific heads.

        Returns:
            torch.Tensor: Attention tensor with shape:
                         - [num_heads, seq_len, seq_len] if head_idx is None
                         - [seq_len, seq_len] if head_idx is an int
                         - [len(head_idx), seq_len, seq_len] if head_idx is a list

        Raises:
            KeyError: If the layer_idx doesn't exist
            IndexError: If head_idx is out of bounds
        """
        if layer_idx not in self.attention_by_layer:
            raise KeyError(f"Layer {layer_idx} not found in attention container")

        attention = self.attention_by_layer[layer_idx]

        if head_idx is None:
            return attention

        # Single head requested
        if isinstance(head_idx, int):
            if head_idx < 0 or head_idx >= attention.shape[0]:
                raise IndexError(
                    f"Head index {head_idx} out of bounds (0-{attention.shape[0]-1})"
                )
            return attention[head_idx]

        # Multiple heads requested
        if isinstance(head_idx, list):
            if not head_idx:
                return attention  # Empty list, return all heads

            max_head = max(head_idx)
            min_head = min(head_idx)
            if min_head < 0 or max_head >= attention.shape[0]:
                raise IndexError(
                    f"Head indices must be between 0-{attention.shape[0]-1}, got {min_head}-{max_head}"
                )

            return attention[head_idx]

        raise TypeError(
            f"head_idx must be None, int, or list of ints, got {type(head_idx)}"
        )

    def has_layer(self, layer_idx: int) -> bool:
        """Check if attention maps for a specific layer exist."""
        return layer_idx in self.attention_by_layer

    def get_layers(self) -> list[int]:
        """Get a list of all layer indices that have attention maps."""
        return sorted(list(self.attention_by_layer.keys()))

    def clear(self):
        """Clear all stored attention maps."""
        self.attention_by_layer.clear()


class LLamaExamineToolkit:
    """
    A toolkit to examine and intervene in LLama model activations.

    Provides utilities for:
      - Identifying key token positions (e.g. newline markers)
      - Computing attention scores
      - Extracting and transplanting activations at specific tokens
    """

    def __init__(self, llama_model, remote=True, num_prev=0):
        """
        Initialize the toolkit.

        Args:
            llama_model: The model instance with attributes like `config` and `tokenizer`.
            remote (bool): Whether to run in remote tracing mode.
        """
        self.llama = llama_model
        self.llama_config = llama_model.config
        self.remote = remote
        self.num_prev = num_prev

    def identify_target_token_index(
        self, string: str, token_string: str= None, index: int = 0
    ) -> tuple[int, int]:
        """
        Identify the token index corresponding to a newline and a cutoff point for the string.

        index tells the index of the token we're looking for. For instance, if there are
        5 instances of the token /n/n,

        The method:
          - Tokenizes the string.
          - Decodes each token individually.
          - Finds the token that contains the maximum number of newline characters.
          - Computes the cutoff index in the original string after decoding tokens up to that point.

        Returns:
            A tuple (newline_token_index, cutoff_char_index) where:
              - newline_token_index: index in the tokenized sequence with the most newlines.
              - cutoff_char_index: character index such that the substring up to that point
                ends exactly after the newline token.
        """
        # Encode the string into tokens
        tokens = self.llama.tokenizer.encode(string)
        # Decode each token individually to inspect its content
        decoded_tokens = [self.llama.tokenizer.decode([token]) for token in tokens]

        #if token_string[]
        # Identify the token index with the maximum newline characters
        newline_index = max(
            range(len(decoded_tokens)), key=lambda i: decoded_tokens[i].count("\n")
        )
        newline_token = decoded_tokens[newline_index]
        # now we identify the indices of tokens here
        newline_index = [i for i, v in enumerate(decoded_tokens) if v == newline_token][
            index
        ]
        # Compute the character cutoff after the newline token (skipping the first token)
        newline_cutoff = len(self.llama.tokenizer.decode(tokens[1 : newline_index + 1]))

        return newline_index, newline_cutoff

    def compute_llama_attention(
        self,
        queries: torch.Tensor,  # Shape: (batch, length, 4096)
        keys: torch.Tensor,  # Shape: (batch, length, 1024)
        average_heads: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention scores with proper rotary position embeddings.

        Args:
            queries: Tensor with shape (batch, seq_length, query_dim).
            keys: Tensor with shape (batch, seq_length, key_dim).
            average_heads: Whether to average over heads in the final output.

        Returns:
            The attention scores tensor.
        """
        if isinstance(queries, np.ndarray):
            queries = torch.tensor(queries)
        if isinstance(queries, np.ndarray):
            keys = torch.tensor(keys)
        # Retrieve head parameters from config
        query_heads = self.llama_config.num_attention_heads
        key_heads = self.llama_config.num_key_value_heads
        head_dim = self.llama_config.head_dim
        batch_size = queries.shape[0]
        seq_length = queries.shape[1]

        # Initialize rotary embeddings with the config
        rotary_emb = LlamaRotaryEmbedding(config=self.llama_config)

        # Generate position IDs for the sequence
        position_ids = (
            torch.arange(0, seq_length, dtype=torch.long, device=queries.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )  # [batch_size, seq_length]

        # Generate cos and sin embeddings
        cos, sin = rotary_emb(queries, position_ids)

        # Reshape queries: (batch, seq_length, query_heads, head_dim)
        queries_reshaped = queries.view(batch_size, seq_length, query_heads, head_dim)

        # Reshape keys: (batch, seq_length, key_heads, head_dim)
        keys_reshaped = keys.view(batch_size, seq_length, key_heads, head_dim)

        # Transpose to (batch, heads, seq_len, head_dim) for rotary embedding
        queries_transposed = queries_reshaped.transpose(1, 2)
        keys_transposed = keys_reshaped.transpose(1, 2)

        # Apply rotary position embeddings
        # Note: unsqueeze_dim=1 is correct when the shape is [batch, heads, seq, dim]
        queries_rotated, keys_rotated = apply_rotary_pos_emb(
            queries_transposed, keys_transposed, cos, sin, unsqueeze_dim=1
        )

        # Repeat keys along head dimension if needed
        multiplier = query_heads // key_heads
        if multiplier > 1:
            keys_rotated = keys_rotated.repeat_interleave(multiplier, dim=1)

        # Compute attention scores with Einstein summation
        # (batch, heads, query_len, key_len)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", queries_rotated, keys_rotated)

        # Create and apply causal mask
        causal_mask = (
            torch.tril(
                torch.ones(
                    (seq_length, seq_length),
                    dtype=torch.bool,
                    device=attn_scores.device,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        attn_scores.masked_fill_(~causal_mask, torch.finfo(attn_scores.dtype).min)

        # Scale and normalize
        attn_norm = torch.softmax(attn_scores / math.sqrt(head_dim), dim=-1)

        # Optionally average over heads
        if average_heads:
            return torch.mean(attn_norm, dim=1)
        return attn_norm

    def produce_attention_distributions(self, strings: list[str]) -> list[torch.Tensor]:
        """
        Given a list of strings, evaluate the attention across the strings
        """
        attention_containers = [AttentionContainer(string) for string in strings]
        raw_qks = defaultdict(list)
        with self.llama.trace(remote=True) as tracer:
            for string_idx, string in enumerate(strings):
                with tracer.invoke(string):
                    # Extract the output activation at the newline token for each layer
                    for layer_idx, layer in enumerate(self.llama.model.layers):
                        layer_q = layer.self_attn.q_proj.output.save()
                        layer_k = layer.self_attn.k_proj.output.save()

                        raw_qks[string].append((layer_q, layer_k))

        for string_idx, string in enumerate(strings):
            for layer_idx, layer in enumerate(self.llama.model.layers):
                layer_q, layer_k = raw_qks[string][layer_idx]
                cur_attention = self.compute_llama_attention(
                    queries=layer_q, keys=layer_k
                )

                attention_containers[string_idx].set_attention(layer_idx, cur_attention)

        return attention_containers

    def produced_plotted_attentions(
        self,
        strings: list[str],
        attention_containers: list[AttentionContainer] | None = None,
        output_directory: str = "attention_plots",  # Ensure it's relative to current dir
    ):
        """
        Save plots of attentions.

        Saves each attention matrix according to the pattern:
        {output_directory}/{string}/layer_{layer_idx}_head_{head_idx}.png

        For each string, we create a subdirectory named after the string (with non-alphanumeric
        characters replaced by underscores). Each index on the attention plot is labeled by
        its token index and token text (rotated 90 degrees on the x-axis and horizontal on the y-axis).
        """
        if attention_containers is None:
            attention_containers = self.produce_attention_distributions(strings)

        # Ensure output directory is absolute and relative to the current working directory
        output_directory = os.path.join(os.getcwd(), output_directory)
        output_directory = os.path.abspath(output_directory)  # Normalize path
        os.makedirs(output_directory, exist_ok=True)

        for string_idx, string in enumerate(strings):
            # Sanitize the string for a folder name
            now = datetime.now()
            datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            sanitized_string = "".join([c if c.isalnum() else "_" for c in string])[:20]
            final_directory_name = f"{sanitized_string}_{datetime_str}"

            # Create a valid absolute directory path
            string_subdir = os.path.join(output_directory, final_directory_name)
            string_subdir = os.path.abspath(string_subdir)  # Normalize path
            os.makedirs(string_subdir, exist_ok=True)

            # Tokenize and decode each token (for axis labels)
            token_ids = self.llama.tokenizer.encode(string)
            token_labels = [self.llama.tokenizer.decode([tid]) for tid in token_ids]
            seq_len = len(token_ids)

            # Retrieve the corresponding AttentionContainer
            container = attention_containers[string_idx]

            # Plot each layer and head
            for layer_idx in container.get_layers():
                # Full attention for this layer: shape [num_heads, seq_len, seq_len]
                layer_attention = container.get_attention(layer_idx)

                # Number of heads
                num_heads = layer_attention.shape[0]

                for head_idx in range(num_heads):
                    # Extract attention for this head
                    if isinstance(layer_attention, torch.Tensor):
                        head_attn = (
                            layer_attention[head_idx].float().detach().cpu().numpy()
                        )
                    else:
                        head_attn = layer_attention[head_idx]

                    # Create a new figure for each head
                    plt.figure(
                        figsize=(16, 16)
                    )  # Double the typical default figure size
                    plt.imshow(head_attn, aspect="auto")

                    # Label ticks with "index token"
                    indices = range(seq_len)
                    xtick_labels = [f"{i} {token_labels[i]}" for i in indices]
                    ytick_labels = [f"{i} {token_labels[i]}" for i in indices]

                    plt.xticks(indices, xtick_labels, rotation=90)
                    plt.yticks(indices, ytick_labels)
                    plt.title(f"Layer {layer_idx}, Head {head_idx}")
                    plt.colorbar()

                    # Optionally, ensure there is enough padding around edges
                    plt.tight_layout()  # Adjusts spacing so x and y labels don't overlap the figure edges

                    outpath = os.path.join(
                        string_subdir, f"layer_{layer_idx}_head_{head_idx}.png"
                    )
                    plt.savefig(outpath, bbox_inches="tight")
                    plt.close()

    def extract_newline_activations(
        self,
        strings: list[str],
        index: int = 0,
        transplant_strings: tuple[str] = ("key", "value"),
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Extract activations at the newline token position for a list of strings.

        For each input string, the method:
          1. Identifies the newline token index and the corresponding cutoff.
          2. Traces the model with the substring up to the newline.
          3. Saves and collects the activations from the keys and values of the self-attention block.

        Args:
            strings: A list of input strings.
            index: the index of the newline we'll be taking.
                by default this is 0, but it could be any newline.

        Returns:
            A list (for each string) of lists (for each layer) of activation tensors.
        """
        print("extracting newline activations")
        # Compute token indices and cutoff positions for all strings
        index_pairs = [
            self.identify_target_token_index(string, index=index) for string in strings
        ]
        activation_containers = [ActivationContainer() for string in strings]

        source_token_indices = []

        # Start tracing for activation capture
        with self.llama.trace(remote=self.remote) as tracer:
            for string, (token_idx, cutoff_idx), ac in zip(
                strings, index_pairs, activation_containers
            ):
                cutoff_string = string[:cutoff_idx]
                source_token_indices.append(token_idx)
                with tracer.invoke(cutoff_string):
                    # Extract the output activation at the newline token for each layer
                    for layer_idx, layer in enumerate(self.llama.model.layers):
                        for delta in range(self.num_prev + 1):
                            if "key" in transplant_strings:
                                ac.set_activation(
                                    token_index=token_idx - delta,
                                    layer_index=layer_idx,
                                    tensor=layer.self_attn.k_proj.output[
                                        :, token_idx - delta, :
                                    ].save(),
                                    label="key",
                                )
                            if "value" in transplant_strings:
                                ac.set_activation(
                                    token_index=token_idx - delta,
                                    layer_index=layer_idx,
                                    tensor=layer.self_attn.v_proj.output[
                                        :, token_idx - delta, :
                                    ].save(),
                                    label="value",
                                )
                            if "output" in transplant_strings:
                                ac.set_activation(
                                    token_index=token_idx - delta,
                                    layer_index=layer_idx,
                                    tensor=layer.self_attn.o_proj.output[
                                        :, token_idx - delta, :
                                    ].save(),
                                    label="output",
                                )
                            # Capture residual stream at key points
                            if "residual" in transplant_strings:
                                # input to self-attention
                                ac.set_activation(
                                    token_index=token_idx - delta,
                                    layer_index=layer_idx,
                                    tensor=layer.input[:, token_idx - delta, :].save(),
                                    label="residual_input",
                                )
                                if layer_idx == self.llama_config.num_hidden_layers - 1:
                                    ac.set_activation(
                                        token_index=token_idx - delta,
                                        layer_index=layer_idx,
                                        tensor=self.llama.model.norm.output[
                                            :, token_idx - delta, :
                                        ].save(),
                                        label="final_residual_output",
                                    )

                            token_int = self.llama.tokenizer.encode(cutoff_string)[
                                token_idx - delta
                            ]
                            ac.set_tokens(
                                token_index=token_idx - delta,
                                token_int=token_int,
                                token_string=self.llama.tokenizer.decode(token_int),
                            )

        return activation_containers, source_token_indices

    def generate_with_transplanted_activity(
        self,
        target_string: str,
        source_token_index: int,
        activation_container: ActivationContainer,
        num_new_tokens: int,
        index: int = 0,
        transplant_strings: tuple[str] = ("key", "value"),
    ):
        """
        Generate new tokens from an input string while intervening at the newline token.

        This method:
          1. Identifies the newline token in the string.
          2. Truncates the string at that point.
          3. During generation, replaces the activation at the newline token in the first new token
             with the provided key, value activities (a tuple of key, value tensors, one per layer).

        Args:
            string: The input string to generate from.
            activities: A list of activation tensors, one per layer.
            num_new_tokens: The number of tokens to generate.

        Returns:
            The raw generated tokens.
        """
        print("generating with transplant")
        layers = self.llama.model.layers
        num_layers = self.llama_config.num_hidden_layers
        target_token_idx, cutoff_idx = self.identify_target_token_index(
            string=target_string, index=index
        )
        cutoff_string = target_string[:cutoff_idx]

        with self.llama.generate(
            cutoff_string,
            max_new_tokens=num_new_tokens,
            remote=self.remote,
        ) as tracer:
            # Intervene on the first generation step only
            for idx in range(num_new_tokens):
                if idx == 0:
                    # Replace activations at the newline token across all layers
                    for i in range(num_layers):
                        for delta in range(self.num_prev + 1):
                            if "key" in transplant_strings:
                                layers[i].self_attn.k_proj.output[
                                    :, target_token_idx - delta, :
                                ] = activation_container.get_activation(
                                    source_token_index - delta, i, "key"
                                )
                            if "value" in transplant_strings:
                                layers[i].self_attn.v_proj.output[
                                    :, target_token_idx - delta, :
                                ] = activation_container.get_activation(
                                    source_token_index - delta, i, "value"
                                )
                            if "output" in transplant_strings:
                                layers[i].self_attn.o_proj.output[
                                    :, target_token_idx - delta, :
                                ] = activation_container.get_activation(
                                    source_token_index - delta, i, "output"
                                )
                            if "residual" in transplant_strings:
                                # 1. Beginning of layer (input residual stream)
                                layers[i].input[:, target_token_idx - delta, :] = (
                                    activation_container.get_activation(
                                        source_token_index - delta, i, "residual_input"
                                    )
                                )
                                if i == self.llama_config.num_hidden_layers - 1:
                                    self.llama.model.norm.output[
                                        :, target_token_idx - delta, :
                                    ] = activation_container.get_activation(
                                        token_index=source_token_index - delta,
                                        layer_index=i,
                                        label="final_residual_output",
                                    )

                            print(
                                "source_token = ",
                                target_token_idx - delta,
                                activation_container.get_token_by_index(
                                    source_token_index - delta
                                ),
                            )
                            toks = self.llama.tokenizer.encode(cutoff_string)
                            print(
                                "target_token = ",
                                target_token_idx - delta,
                                (
                                    self.llama.tokenizer.decode(
                                        toks[target_token_idx - delta]
                                    ),
                                ),
                            )

            # Proceed to the next token generation step
            layers.next()

            # Save and retrieve the generator output tokens
            out = self.llama.generator.output.save()
        return out

    def transplant_newline_activities(
        self,
        source_strings: list[str],
        target_strings: list[str],
        num_new_tokens: int,
        index: int = 0,
        transplant_strings: tuple[str] = ("key", "value"),
    ) -> list[str]:
        """
        Transplant the newline token activations from source strings to target strings.
        This applies it to a particular newline token, given by index.

        For each corresponding source and target string pair, the method:
          1. Extracts newline activations from the source.
          2. Generates new tokens for the target while intervening to replace the newline token
             activation with that from the source.

        Note: This method currently requires exactly two strings for both source and target.

        Args:
            source_strings: A list of source strings (length must be 2).
            target_strings: A list of target strings (length must be 2).
            num_new_tokens: The number of tokens to generate for each target string.

        Returns:
            A list of generated strings for each target.
        """
        # Extract newline activations from source strings
        activation_containers, source_newline_indices = (
            self.extract_newline_activations(
                strings=source_strings,
                index=index,
                transplant_strings=transplant_strings,
            )
        )
        output_strings = []

        # Process each target string with corresponding source activations
        for target_string, activation_container, source_newline_index in zip(
            target_strings, activation_containers, source_newline_indices
        ):
            tokens = self.generate_with_transplanted_activity(
                target_string=target_string,
                activation_container=activation_container,
                source_token_index=source_newline_index,
                num_new_tokens=num_new_tokens,
                index=index,
                transplant_strings=transplant_strings,
            )
            decoded = self.llama.tokenizer.decode(tokens[0])
            output_strings.append(decoded)

        self.activation_containers = activation_containers

        return output_strings
