import math
import torch
import matplotlib.pyplot as plt
from nnsight import CONFIG
import nnsight
from dataclasses import dataclass
from collections import defaultdict


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


class LLamaExamineToolkit:
    """
    A toolkit to examine and intervene in LLama model activations.

    Provides utilities for:
      - Identifying key token positions (e.g. newline markers)
      - Computing attention scores
      - Extracting and transplanting activations at specific tokens
    """

    def __init__(self, llama_model, remote=True, num_prev=1):
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

    def identify_newline_index(self, string: str, index: int = 0) -> tuple[int, int]:
        """
        Identify the token index corresponding to a newline and a cutoff point for the string.

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
        queries_out: torch.Tensor,
        keys_out: torch.Tensor,
        average_heads: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention scores using a simplified, intervention-friendly approach.

        This method reshapes the queries and keys to separate heads, computes
        dot-product attention with a causal mask, scales the scores, and applies softmax.
        Optionally, it averages the attention scores over heads.

        Args:
            queries_out: Tensor with shape (batch, seq_length, query_dim).
            keys_out: Tensor with shape (batch, seq_length, key_dim).
            average_heads: Whether to average over heads in the final output.

        Returns:
            The attention scores tensor, with shape:
              - (batch, num_heads, seq_length, seq_length) if average_heads is False, or
              - (batch, seq_length, seq_length) if averaged.
        """
        # Retrieve head parameters from config
        query_heads = self.llama_config.num_attention_heads
        key_heads = self.llama_config.num_key_value_heads
        head_dim = self.llama_config.head_dim
        batch_size = queries_out.shape[0]

        # Determine sequence length; note: avoid hardcoding if possible
        seq_length = queries_out.shape[1]

        # Reshape queries: (batch, seq_length, query_heads, head_dim)
        queries_out = queries_out.view(batch_size, seq_length, query_heads, head_dim)

        # Reshape keys: (batch, seq_length, key_heads, head_dim)
        keys_out = keys_out.view(batch_size, seq_length, key_heads, head_dim)
        # Repeat keys along head dimension to match query_heads (assumes query_heads % key_heads == 0)
        multiplier = query_heads // key_heads
        keys_out = keys_out.repeat_interleave(multiplier, dim=2)

        # Compute attention scores with Einstein summation:
        # Resulting shape: (batch, num_heads, seq_length (queries), seq_length (keys))
        attn_scores = torch.einsum("bLnd,blnd->bnLl", queries_out, keys_out)

        # Create a causal mask so that each token can only attend to previous tokens
        causal_mask = torch.tril(
            torch.ones(
                (seq_length, seq_length), dtype=torch.bool, device=attn_scores.device
            )
        )
        # Reshape for broadcasting: (1, 1, seq_length, seq_length)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply the causal mask: set positions that should not be attended to a very low value
        attn_scores.masked_fill_(~causal_mask, torch.finfo(attn_scores.dtype).min)

        # Scale and normalize the attention scores using softmax
        attn_norm = torch.softmax(attn_scores / math.sqrt(head_dim), dim=-1)

        # Optionally average over heads (dim=1)
        if average_heads:
            return torch.mean(attn_norm, dim=1)
        return attn_norm

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
            self.identify_newline_index(string, index=index) for string in strings
        ]
        activation_containers = [ActivationContainer() for string in strings]

        # Start tracing for activation capture
        with self.llama.trace(remote=self.remote) as tracer:
            for string, (token_idx, cutoff_idx), ac in zip(
                strings, index_pairs, activation_containers
            ):
                cutoff_string = string[:cutoff_idx]
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
                            token_int = self.llama.tokenizer.encode(cutoff_string)[
                                token_idx - delta
                            ]
                            ac.set_tokens(
                                token_index=token_idx - delta,
                                token_int=token_int,
                                token_string=self.llama.tokenizer.decode(token_int),
                            )

        return activation_containers

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
        layers = self.llama.model.layers
        num_layers = self.llama_config.num_hidden_layers
        target_token_idx, cutoff_idx = self.identify_newline_index(
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
                            print(
                                "target_token = ",
                                target_token_idx - delta,
                                activation_container.get_token_by_index(
                                    target_token_idx - delta
                                ),
                            )
                            toks = self.llama.tokenizer.encode(cutoff_string)
                            print(
                                "source_token = ",
                                source_token_index - delta,
                                self.llama.tokenizer.decode(
                                    toks[source_token_index - delta]
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
        activation_containers = self.extract_newline_activations(
            strings=source_strings, index=index, transplant_strings=transplant_strings
        )
        output_strings = []

        # Process each target string with corresponding source activations
        for target_string, activation_container in zip(
            target_strings, activation_containers
        ):
            tokens = self.generate_with_transplanted_activity(
                target_string=target_string,
                activation_container=activation_container,
                num_new_tokens=num_new_tokens,
                index=index,
                transplant_strings=transplant_strings,
            )
            decoded = self.llama.tokenizer.decode(tokens[0])
            output_strings.append(decoded)

        self.activation_containers = activation_containers

        return output_strings
