"""Utilities for generating short English word lists and context-stream lexicons."""

from .generator import WordLists, generate_word_lists, write_word_lists
from .lexicon import (
    DomainAuditRow,
    DomainDefinition,
    LexiconWord,
    audit_domains,
    build_domain_lexicon,
    build_foundation_lexicon,
    lexicon_entries_from_json,
    matching_words,
    morse_for_word,
    read_lexicon_asset,
    rhythm_diversity_for_word,
    rhythm_metadata_for_word,
    rhythm_signature_for_word,
    selectable_words,
)

__all__ = [
    "DomainAuditRow",
    "DomainDefinition",
    "LexiconWord",
    "WordLists",
    "audit_domains",
    "build_domain_lexicon",
    "build_foundation_lexicon",
    "generate_word_lists",
    "lexicon_entries_from_json",
    "matching_words",
    "morse_for_word",
    "read_lexicon_asset",
    "rhythm_diversity_for_word",
    "rhythm_metadata_for_word",
    "rhythm_signature_for_word",
    "selectable_words",
    "write_word_lists",
]
