"""Vocabulary class for symbolic regression."""

import json
from typing import List, Set


class Vocabulary:
    """Vocabulary for symbolic regression with operators, functions, variables, and special tokens."""

    # Define token categories as class constants for reference
    OPERATORS = ['add', 'sub', 'mul', 'div', 'pow']
    FUNCTIONS = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'arcsin', 'arccos', 'arctan']
    SPECIAL_TOKENS = ['<gap>', 'constant', '<s>', '</s>', '<pad>', '<mask>']

    def __init__(self, num_variables: int = 0, data_vocab_size: int = None):
        """Initialize the vocabulary with all tokens.

        Args:
            num_variables: Number of variable tokens to add (x0, x1, x2, ...)
            data_vocab_size: Optional vocab size for data format (if different from vocab_size)
        """
        # Define all tokens in order: operators, functions, variables, special tokens
        self._operators = list(self.OPERATORS)
        self._functions = list(self.FUNCTIONS)
        self._variables = [f'x{i}' for i in range(num_variables)]
        self._special_tokens = list(self.SPECIAL_TOKENS)

        # Build token list in order for ID assignment
        self._tokens = self._operators + self._functions + self._variables + self._special_tokens

        # Build token2id and id2token dictionaries
        self._token_to_id = {token: idx for idx, token in enumerate(self._tokens)}
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

        # Store data_vocab_size for compatibility with data using different token mappings
        self._data_vocab_size = data_vocab_size

    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return len(self._tokens)

    @property
    def data_vocab_size(self) -> int:
        """Return the data vocab size (for one-hot encoding with data compatibility)."""
        return self._data_vocab_size if self._data_vocab_size is not None else self.vocab_size

    @property
    def special_tokens(self) -> Set[str]:
        """Return the set of special tokens."""
        return set(self._special_tokens)

    @property
    def operator_tokens(self) -> Set[str]:
        """Return the set of operator tokens."""
        return set(self._operators)

    @property
    def function_tokens(self) -> Set[str]:
        """Return the set of function tokens."""
        return set(self._functions)

    @property
    def variable_tokens(self) -> Set[str]:
        """Return the set of variable tokens."""
        return set(self._variables)

    @property
    def gap_token(self) -> int:
        """Return the gap token ID."""
        return self.token_to_id('<gap>')

    @property
    def pad_token(self) -> int:
        """Return the pad token ID."""
        return self.token_to_id('<pad>')

    def token_to_id(self, token: str) -> int:
        """Convert token string to integer ID."""
        if token not in self._token_to_id:
            raise KeyError(f"Token '{token}' not found in vocabulary.")
        return self._token_to_id[token]

    def id_to_token(self, id: int) -> str:
        """Convert integer ID to token string."""
        if id not in self._id_to_token:
            raise IndexError(f"ID {id} out of range for vocabulary of size {self.vocab_size}.")
        return self._id_to_token[id]

    def __contains__(self, token: str) -> bool:
        """Check if token exists in vocabulary."""
        return token in self._token_to_id

    def encode(self, tokens: List[str]) -> List[int]:
        """Encode list of tokens to list of IDs."""
        return [self.token_to_id(token) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Decode list of IDs to list of tokens."""
        return [self.id_to_token(idx) for idx in ids]

    def is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        return token in self._special_tokens

    def is_operator(self, token: str) -> bool:
        """Check if token is an operator."""
        return token in self._operators

    def is_function(self, token: str) -> bool:
        """Check if token is a function."""
        return token in self._functions

    def is_variable(self, token: str) -> bool:
        """Check if token is a variable."""
        return token in self._variables

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            'operators': self._operators,
            'functions': self._functions,
            'variables': self._variables,
            'special_tokens': self._special_tokens,
            'token_to_id': self._token_to_id,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        vocab = cls()
        # Override the loaded data to match saved state
        vocab._operators = data['operators']
        vocab._functions = data['functions']
        vocab._variables = data['variables']
        vocab._special_tokens = data['special_tokens']
        vocab._tokens = vocab._operators + vocab._functions + vocab._variables + vocab._special_tokens
        vocab._token_to_id = data['token_to_id']
        vocab._id_to_token = {int(idx): token for token, idx in vocab._token_to_id.items()}
        return vocab
