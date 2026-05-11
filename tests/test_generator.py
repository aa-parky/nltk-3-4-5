from pathlib import Path

import pytest

from nltk_3_4_5.generator import (
    generate_word_lists,
    normalise_allowed_letters,
    word_is_eligible,
    write_word_lists,
)


def test_normalise_allowed_letters_accepts_letters_only() -> None:
    assert normalise_allowed_letters("A B-C1") == frozenset({"a", "b", "c"})


def test_normalise_allowed_letters_rejects_non_alphabetic_filter() -> None:
    with pytest.raises(ValueError, match="at least one alphabetic"):
        normalise_allowed_letters("123---")


def test_word_is_eligible_excludes_capitalised_words_by_default() -> None:
    assert word_is_eligible("cat", length=3)
    assert not word_is_eligible("Cat", length=3)
    assert word_is_eligible("Cat", length=3, include_proper_nouns=True)


def test_word_is_eligible_requires_exact_length_and_letters() -> None:
    assert word_is_eligible("home", length=4)
    assert not word_is_eligible("home", length=3)
    assert not word_is_eligible("can't", length=5)


def test_generate_word_lists_groups_sorted_unique_lowercase_words() -> None:
    corpus = ["cat", "dog", "dog", "deer", "bear", "apple", "Zed", "can't"]

    generated = generate_word_lists(corpus_words=corpus)

    assert generated[3] == ("cat", "dog")
    assert generated[4] == ("bear", "deer")
    assert generated[5] == ("apple",)


def test_generate_word_lists_can_filter_to_allowed_letters() -> None:
    corpus = ["tea", "eat", "ate", "tan", "team", "meat", "mate", "steam"]

    generated = generate_word_lists(corpus_words=corpus, allowed_letters="aemt")

    assert generated[3] == ("ate", "eat", "tea")
    assert generated[4] == ("mate", "meat", "team")
    assert generated[5] == ()


def test_write_word_lists_creates_one_file_per_length(tmp_path: Path) -> None:
    generated = generate_word_lists(corpus_words=["cat", "wolf", "zebra"])

    files = write_word_lists(generated, tmp_path)

    assert files[3].read_text(encoding="utf-8") == "cat\n"
    assert files[4].read_text(encoding="utf-8") == "wolf\n"
    assert files[5].read_text(encoding="utf-8") == "zebra\n"
