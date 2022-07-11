from pathlib import Path
from typing import Iterable
from typing import Union

from typeguard import check_argument_types

from text.abs_tokenizer import AbsTokenizer
from text.char_tokenizer import CharTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer
    NOTE(kinouchi):In Japanese, tokenizer is fixed Character
    """
    assert check_argument_types()

    return CharTokenizer(
        non_linguistic_symbols=non_linguistic_symbols,
        space_symbol=space_symbol,
        remove_non_linguistic_symbols=remove_non_linguistic_symbols,
    )
