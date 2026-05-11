"""Core word-list generation logic.

The module deliberately keeps corpus access separate from filtering and writing so that the
behaviour remains easy to test and adapt. By default it uses NLTK's ``words`` corpus because
that corpus is a broad English word list suitable for simple spelling-length filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import nltk
from nltk.corpus import words as nltk_words
from nltk.data import find

DEFAULT_LENGTHS = (3, 4, 5)
DEFAULT_CORPUS = "words"


@dataclass(frozen=True)
class WordLists:
    """Generated word lists grouped by word length."""

    by_length: Mapping[int, tuple[str, ...]]

    def __getitem__(self, length: int) -> tuple[str, ...]:
        """Return the generated words for a specific length."""

        return self.by_length[length]


def ensure_corpus(corpus_name: str = DEFAULT_CORPUS) -> None:
    """Ensure that the requested NLTK corpus is available locally.

    NLTK does not always install corpora with the Python package itself. The first run can
    therefore download the small ``words`` corpus automatically, while later runs use the
    cached copy.
    """

    try:
        find(f"corpora/{corpus_name}")
    except LookupError:
        nltk.download(corpus_name, quiet=True)


def normalise_allowed_letters(letters: str | None) -> frozenset[str] | None:
    """Return a lowercase set of allowed letters, or ``None`` if no filter is requested."""

    if not letters:
        return None

    allowed = frozenset(character.lower() for character in letters if character.isalpha())
    if not allowed:
        msg = "The allowed letters filter must contain at least one alphabetic character."
        raise ValueError(msg)
    return allowed


def word_is_eligible(
    word: str,
    *,
    length: int,
    allowed_letters: frozenset[str] | None = None,
    include_proper_nouns: bool = False,
) -> bool:
    """Decide whether a corpus token should appear in one of the output lists."""

    if len(word) != length:
        return False

    if not word.isalpha():
        return False

    if not include_proper_nouns and word != word.lower():
        return False

    lowered = word.lower()
    if allowed_letters is not None and not set(lowered).issubset(allowed_letters):
        return False

    return True


def generate_word_lists(
    *,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    allowed_letters: str | None = None,
    include_proper_nouns: bool = False,
    corpus_words: Iterable[str] | None = None,
) -> WordLists:
    """Generate English word lists grouped by exact word length.

    Args:
        lengths: Exact word lengths to generate. The default is ``3, 4, 5``.
        allowed_letters: Optional alphabetic character set. When supplied, generated words
            must be composed only from these letters.
        include_proper_nouns: Include capitalised corpus entries when set to ``True``.
        corpus_words: Optional iterable for testing or advanced custom generation. When not
            supplied, the function reads from NLTK's ``words`` corpus.

    Returns:
        A ``WordLists`` object containing sorted, unique, lowercase words for each length.
    """

    target_lengths = tuple(dict.fromkeys(int(length) for length in lengths))
    if any(length < 1 for length in target_lengths):
        msg = "All requested word lengths must be positive integers."
        raise ValueError(msg)

    allowed = normalise_allowed_letters(allowed_letters)

    if corpus_words is None:
        ensure_corpus(DEFAULT_CORPUS)
        corpus_words = nltk_words.words()

    grouped: dict[int, set[str]] = {length: set() for length in target_lengths}
    for word in corpus_words:
        for length in target_lengths:
            if word_is_eligible(
                word,
                length=length,
                allowed_letters=allowed,
                include_proper_nouns=include_proper_nouns,
            ):
                grouped[length].add(word.lower())

    return WordLists(
        by_length={length: tuple(sorted(words)) for length, words in grouped.items()}
    )


def write_word_lists(word_lists: WordLists, output_directory: str | Path = "output") -> dict[int, Path]:
    """Write each generated list to a separate newline-delimited text file."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[int, Path] = {}
    for length, words in word_lists.by_length.items():
        target = output_path / f"{length}_letter_words.txt"
        target.write_text("\n".join(words) + ("\n" if words else ""), encoding="utf-8")
        written_files[length] = target

    return written_files
