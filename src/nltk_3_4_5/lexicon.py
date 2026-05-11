"""Domain-aware lexicon generation for Morse context streams.

This module builds offline JSON assets that can be consumed by the Copy app without
requiring linguistic processing at runtime. It treats NLTK as a candidate source and
scoring aid, then exports small, deterministic, metadata-rich word records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping
import json

import nltk
from nltk.corpus import brown, wordnet as wn, words as nltk_words
from nltk.data import find

from .generator import DEFAULT_LENGTHS, ensure_corpus, normalise_allowed_letters

FREQUENCY_CORPUS = "brown"
WORDNET_CORPUS = "wordnet"

DEFAULT_EARLY_KOCH_SEQUENCE = "kmuresnaptlwiojzfd yvg5/q9zh38b?47c1d60x".replace(" ", "")
FOUNDATION_TAG = "foundation"

MORSE_CODE: dict[str, str] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}


@dataclass(frozen=True)
class DomainDefinition:
    """A domain that may produce useful short-word context streams."""

    key: str
    label: str
    seeds: tuple[str, ...]
    curated_words: tuple[str, ...] = ()
    cue_templates: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LexiconWord:
    """A JSON-ready word record for app-side stream selection."""

    word: str
    length: int
    letters: tuple[str, ...]
    tags: tuple[str, ...]
    frequency: int
    frequency_rank: int | None
    commonness: str
    domain_scores: Mapping[str, float]
    morse: str = ""
    rhythm_signature: str = ""
    dit_count: int = 0
    dah_count: int = 0
    transitions: int = 0
    repeat_pressure: str = "low"
    rhythm_diversity: float = 0.0


@dataclass(frozen=True)
class DomainAuditRow:
    """Summary statistics for deciding whether a domain is worth keeping."""

    domain: str
    label: str
    total_words: int
    three_letter: int
    four_letter: int
    five_letter: int
    common_ratio: float
    early_character_yield: int
    average_domain_score: float
    verdict: str


DEFAULT_DOMAINS: tuple[DomainDefinition, ...] = (
    DomainDefinition(
        key="animals",
        label="Animals",
        seeds=("animal", "mammal", "bird", "fish", "insect", "pet"),
        curated_words=(
            "ape", "bat", "cat", "cow", "dog", "eel", "emu", "fox", "hen", "hog",
            "owl", "pig", "rat", "yak", "bear", "bird", "calf", "crab", "deer", "duck",
            "fish", "foal", "frog", "goat", "hare", "hawk", "lamb", "lion", "mole", "seal",
            "swan", "toad", "wolf", "zebu", "camel", "eagle", "goose", "horse", "mouse",
            "otter", "panda", "shark", "sheep", "snail", "tiger", "whale", "zebra",
        ),
    ),
    DomainDefinition(
        key="food",
        label="Food",
        seeds=("food", "fruit", "vegetable", "meal", "meat", "drink"),
        curated_words=(
            "ale", "bun", "egg", "fig", "ham", "jam", "nut", "oat", "pea", "pie", "rye",
            "tea", "yam", "bean", "beef", "beer", "cake", "corn", "date", "fish", "kiwi",
            "lamb", "lime", "loaf", "meat", "milk", "pear", "plum", "pork", "rice", "soup",
            "tart", "veal", "apple", "bacon", "bread", "cream", "grape", "honey", "lemon",
            "melon", "olive", "onion", "peach", "salad", "sauce", "steak", "sugar",
        ),
    ),
    DomainDefinition(
        key="home",
        label="Home",
        seeds=("home", "house", "room", "furniture", "tool", "clothing"),
        curated_words=(
            "bed", "box", "cup", "fan", "jar", "key", "mat", "mug", "pan", "pot", "rug",
            "tap", "tin", "bag", "bath", "bell", "bolt", "book", "bowl", "coat", "door",
            "fork", "lamp", "lock", "nail", "oven", "plug", "ring", "sink", "soap", "sock",
            "wall", "wire", "bench", "chair", "clock", "floor", "glass", "knife", "plate",
            "shelf", "shirt", "table", "towel",
        ),
    ),
    DomainDefinition(
        key="nature",
        label="Nature",
        seeds=("nature", "plant", "tree", "river", "forest", "land"),
        curated_words=(
            "ash", "bay", "bog", "dew", "elm", "fir", "fog", "ice", "ivy", "mud", "oak",
            "sea", "sky", "sun", "air", "bark", "bush", "clay", "fern", "hill", "lake",
            "leaf", "moon", "moss", "rain", "reed", "rock", "root", "sand", "snow", "soil",
            "star", "tree", "wind", "beach", "berry", "field", "grass", "river", "stone",
            "storm", "water", "woods",
        ),
    ),
    DomainDefinition(
        key="weather",
        label="Weather",
        seeds=("weather", "rain", "wind", "storm", "cloud", "snow"),
        curated_words=(
            "fog", "ice", "sun", "air", "cold", "damp", "gale", "hail", "heat", "mist",
            "rain", "snow", "warm", "wind", "cloud", "frost", "sleet", "storm",
        ),
    ),
    DomainDefinition(
        key="maritime",
        label="Maritime",
        seeds=("ship", "boat", "sea", "sail", "harbor", "navigation"),
        curated_words=(
            "bay", "oar", "sea", "ark", "boat", "crew", "deck", "dock", "gulf", "hull",
            "keel", "knot", "mast", "pier", "port", "rope", "sail", "ship", "tide", "wake",
            "water", "wharf",
        ),
    ),
    DomainDefinition(
        key="travel",
        label="Travel",
        seeds=("travel", "road", "vehicle", "journey", "transport"),
        curated_words=(
            "bus", "cab", "car", "jet", "map", "taxi", "bike", "boat", "fare", "ford", "gate",
            "mile", "path", "port", "road", "ship", "tram", "trip", "walk", "wheel", "train",
        ),
    ),
    DomainDefinition(
        key="music",
        label="Music",
        seeds=("music", "instrument", "song", "sound"),
        curated_words=(
            "air", "bar", "key", "rap", "song", "bass", "bell", "drum", "flute", "harp",
            "horn", "lute", "note", "tune", "viola", "voice",
        ),
    ),
    DomainDefinition(
        key="body",
        label="Body",
        seeds=("body", "organ", "limb", "face"),
        curated_words=(
            "arm", "ear", "eye", "leg", "rib", "toe", "back", "bone", "chin", "face", "foot",
            "hair", "hand", "head", "knee", "lung", "neck", "nose", "skin", "tooth", "wrist",
        ),
    ),
    DomainDefinition(
        key="work",
        label="Work",
        seeds=("work", "job", "office", "trade", "craft"),
        curated_words=(
            "job", "pay", "pen", "boss", "desk", "file", "form", "note", "plan", "shop",
            "task", "team", "tool", "trade", "wage", "write",
        ),
    ),
    DomainDefinition(
        key="emergency",
        label="Emergency",
        seeds=("emergency", "rescue", "danger", "fire", "aid"),
        curated_words=(
            "aid", "cut", "ill", "war", "burn", "fire", "help", "hurt", "risk", "safe", "team",
            "alarm", "flood", "guard", "medic", "radio", "rescue", "smoke", "water",
        ),
    ),
    DomainDefinition(
        key="space",
        label="Space",
        seeds=("space", "astronomy", "planet", "star"),
        curated_words=(
            "sun", "sky", "mars", "moon", "star", "comet", "earth", "orbit", "solar", "venus",
        ),
    ),
)


def ensure_lexicon_corpora() -> None:
    """Ensure corpora needed by the offline lexicon builder are available."""

    ensure_corpus("words")
    ensure_corpus(FREQUENCY_CORPUS)
    try:
        find("corpora/wordnet")
    except LookupError:
        nltk.download(WORDNET_CORPUS, quiet=True)


def normalise_word(word: str) -> str | None:
    """Return a clean lowercase single word, or ``None`` if unsuitable."""

    cleaned = word.replace("_", "-").strip().lower()
    if "-" in cleaned or " " in cleaned:
        return None
    if not cleaned.isalpha():
        return None
    return cleaned


def valid_word_set(corpus_words: Iterable[str] | None = None) -> set[str]:
    """Return lower-case alphabetic words accepted as valid English candidates."""

    if corpus_words is None:
        ensure_corpus("words")
        corpus_words = nltk_words.words()
    return {word.lower() for word in corpus_words if word.isalpha() and word == word.lower()}


def brown_frequency() -> tuple[Counter[str], dict[str, int]]:
    """Return Brown-corpus token counts and frequency ranks for alphabetic words."""

    ensure_corpus(FREQUENCY_CORPUS)
    counts: Counter[str] = Counter(
        token.lower() for token in brown.words() if token.isalpha()
    )
    ranks = {word: index + 1 for index, (word, _count) in enumerate(counts.most_common())}
    return counts, ranks


def commonness_for_rank(rank: int | None, frequency: int) -> str:
    """Classify a word by broad frequency usefulness."""

    if rank is None or frequency == 0:
        return "unranked"
    if rank <= 5_000:
        return "common"
    if rank <= 15_000:
        return "familiar"
    return "rare"


def wordnet_domain_candidates(domain: DomainDefinition, *, max_depth: int = 2) -> dict[str, float]:
    """Collect domain candidates from WordNet seed synsets with rough confidence scores."""

    try:
        find("corpora/wordnet")
    except LookupError:
        nltk.download(WORDNET_CORPUS, quiet=True)

    candidates: dict[str, float] = {}

    def add_lemmas(synset: wn.synset, score: float) -> None:
        for lemma in synset.lemma_names():
            word = normalise_word(lemma)
            if word is not None:
                candidates[word] = max(candidates.get(word, 0.0), score)

    for seed in domain.seeds:
        for synset in wn.synsets(seed, pos=wn.NOUN):
            add_lemmas(synset, 0.95)
            frontier = [(child, 1) for child in synset.hyponyms()]
            while frontier:
                child, depth = frontier.pop(0)
                add_lemmas(child, max(0.3, 0.85 - (depth * 0.15)))
                if depth < max_depth:
                    frontier.extend((grandchild, depth + 1) for grandchild in child.hyponyms())

    for word in domain.curated_words:
        normalised = normalise_word(word)
        if normalised is not None:
            candidates[normalised] = 1.0

    return candidates


def morse_for_word(word: str) -> str:
    """Return a space-separated Morse representation for a lowercase word."""

    return " ".join(MORSE_CODE[letter] for letter in word)


def rhythm_signature_for_word(word: str) -> str:
    """Return a compact D/I rhythm signature with underscores between letters."""

    return "_".join(MORSE_CODE[letter].replace("-", "D").replace(".", "I") for letter in word)


def _max_run_length(symbols: str) -> int:
    """Return the longest uninterrupted dit/dah run in a symbol string."""

    if not symbols:
        return 0
    max_run = 1
    current_run = 1
    for previous, current in zip(symbols, symbols[1:]):
        if current == previous:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def repeat_pressure_for_symbols(symbols: str) -> str:
    """Classify how strongly a rhythm repeats the same symbol."""

    max_run = _max_run_length(symbols)
    if max_run >= 4:
        return "high"
    if max_run >= 2:
        return "medium"
    return "low"


def rhythm_diversity_for_word(word: str) -> float:
    """Score rhythmic variety from 0.0 to 1.0 for early Morse imprinting.

    The score rewards a balanced dit/dah mix, changes between dits and dahs,
    different per-letter rhythm shapes, and avoids excessive repeated runs.
    """

    codes = [MORSE_CODE[letter] for letter in word]
    symbols = "".join(codes)
    total = len(symbols)
    if total == 0:
        return 0.0

    dit_count = symbols.count(".")
    dah_count = symbols.count("-")
    balance = 1.0 - (abs(dit_count - dah_count) / total)
    transition_count = sum(1 for previous, current in zip(symbols, symbols[1:]) if previous != current)
    transition_density = transition_count / max(total - 1, 1)
    letter_shape_variety = len(set(codes)) / len(codes)
    run_penalty = max(0.0, 1.0 - ((_max_run_length(symbols) - 1) / max(total - 1, 1)))

    score = (
        (0.35 * balance)
        + (0.30 * transition_density)
        + (0.20 * letter_shape_variety)
        + (0.15 * run_penalty)
    )
    return round(score, 3)


def rhythm_metadata_for_word(word: str) -> dict[str, object]:
    """Build the rhythm metadata exported with each lexicon entry."""

    morse = morse_for_word(word)
    symbols = morse.replace(" ", "")
    return {
        "morse": morse,
        "rhythm_signature": rhythm_signature_for_word(word),
        "dit_count": symbols.count("."),
        "dah_count": symbols.count("-"),
        "transitions": sum(1 for previous, current in zip(symbols, symbols[1:]) if previous != current),
        "repeat_pressure": repeat_pressure_for_symbols(symbols),
        "rhythm_diversity": rhythm_diversity_for_word(word),
    }


def lexicon_word(
    *,
    word: str,
    tags: tuple[str, ...],
    frequency: int,
    frequency_rank: int | None,
    commonness: str,
    domain_scores: Mapping[str, float],
) -> LexiconWord:
    """Create a lexicon word with standard lexical and Morse rhythm metadata."""

    rhythm = rhythm_metadata_for_word(word)
    return LexiconWord(
        word=word,
        length=len(word),
        letters=tuple(sorted(set(word))),
        tags=tags,
        frequency=frequency,
        frequency_rank=frequency_rank,
        commonness=commonness,
        domain_scores=domain_scores,
        morse=str(rhythm["morse"]),
        rhythm_signature=str(rhythm["rhythm_signature"]),
        dit_count=int(rhythm["dit_count"]),
        dah_count=int(rhythm["dah_count"]),
        transitions=int(rhythm["transitions"]),
        repeat_pressure=str(rhythm["repeat_pressure"]),
        rhythm_diversity=float(rhythm["rhythm_diversity"]),
    )


def build_domain_lexicon(
    *,
    domains: Iterable[DomainDefinition] = DEFAULT_DOMAINS,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    corpus_words: Iterable[str] | None = None,
    frequency_counts: Mapping[str, int] | None = None,
    frequency_ranks: Mapping[str, int] | None = None,
    require_dictionary_word: bool = True,
) -> list[LexiconWord]:
    """Build a merged, taggable lexicon across all configured domains."""

    target_lengths = set(int(length) for length in lengths)
    dictionary = valid_word_set(corpus_words) if require_dictionary_word else set()

    if frequency_counts is None or frequency_ranks is None:
        counts, ranks = brown_frequency()
    else:
        counts = Counter(frequency_counts)
        ranks = dict(frequency_ranks)

    merged: dict[str, dict[str, float]] = {}
    for domain in domains:
        candidates = wordnet_domain_candidates(domain)
        for word, score in candidates.items():
            if len(word) not in target_lengths:
                continue
            if require_dictionary_word and word not in dictionary:
                continue
            # WordNet is excellent for candidate discovery, but it can surface archaic,
            # technical, or highly obscure short words. Keep curated words regardless;
            # otherwise require at least one Brown-corpus occurrence as a light
            # commonness signal before exporting the candidate.
            if score < 1.0 and counts.get(word, 0) == 0:
                continue
            merged.setdefault(word, {})[domain.key] = round(score, 3)

    entries: list[LexiconWord] = []
    for word, domain_scores in sorted(merged.items()):
        frequency = int(counts.get(word, 0))
        rank = ranks.get(word)
        entries.append(
            lexicon_word(
                word=word,
                tags=tuple(sorted(domain_scores)),
                frequency=frequency,
                frequency_rank=rank,
                commonness=commonness_for_rank(rank, frequency),
                domain_scores=dict(sorted(domain_scores.items())),
            )
        )

    return entries


def build_foundation_lexicon(
    *,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    corpus_words: Iterable[str] | None = None,
    frequency_counts: Mapping[str, int] | None = None,
    frequency_ranks: Mapping[str, int] | None = None,
    min_frequency: int = 1,
    tag: str = FOUNDATION_TAG,
) -> list[LexiconWord]:
    """Build a broad, non-domain lexicon for early-stage context streams.

    The foundation lexicon keeps the same JSON shape as the domain lexicon but uses a
    single tag. It is intended for early Morse stages where strict themed domains are too
    sparse, while still excluding words that have no Brown-corpus evidence by default.
    """

    target_lengths = set(int(length) for length in lengths)
    dictionary = valid_word_set(corpus_words)

    if frequency_counts is None or frequency_ranks is None:
        counts, ranks = brown_frequency()
    else:
        counts = Counter(frequency_counts)
        ranks = dict(frequency_ranks)

    entries: list[LexiconWord] = []
    for word in sorted(dictionary):
        if len(word) not in target_lengths:
            continue
        frequency = int(counts.get(word, 0))
        if frequency < min_frequency:
            continue
        rank = ranks.get(word)
        entries.append(
            lexicon_word(
                word=word,
                tags=(tag,),
                frequency=frequency,
                frequency_rank=rank,
                commonness=commonness_for_rank(rank, frequency),
                domain_scores={tag: 1.0},
            )
        )
    return entries


def lexicon_entries_from_json(data: Mapping[str, object]) -> list[LexiconWord]:
    """Rebuild lexicon entries from a context_lexicon.json-style dictionary."""

    raw_words = data.get("words", [])
    if not isinstance(raw_words, list):
        raise ValueError("Lexicon JSON must contain a 'words' list.")

    entries: list[LexiconWord] = []
    for raw_entry in raw_words:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Each lexicon word entry must be a JSON object.")
        word = str(raw_entry["word"])
        letters = tuple(str(letter) for letter in raw_entry["letters"])
        tags = tuple(str(tag) for tag in raw_entry["tags"])
        domain_scores_raw = raw_entry.get("domain_scores", {})
        if not isinstance(domain_scores_raw, Mapping):
            raise ValueError(f"Word {word!r} has invalid domain_scores data.")
        frequency_rank_raw = raw_entry.get("frequency_rank")
        rhythm = rhythm_metadata_for_word(word)
        entries.append(
            LexiconWord(
                word=word,
                length=int(raw_entry["length"]),
                letters=letters,
                tags=tags,
                frequency=int(raw_entry.get("frequency", 0)),
                frequency_rank=None if frequency_rank_raw is None else int(frequency_rank_raw),
                commonness=str(raw_entry.get("commonness", "unranked")),
                domain_scores={str(key): float(value) for key, value in domain_scores_raw.items()},
                morse=str(raw_entry.get("morse", rhythm["morse"])),
                rhythm_signature=str(raw_entry.get("rhythm_signature", rhythm["rhythm_signature"])),
                dit_count=int(raw_entry.get("dit_count", rhythm["dit_count"])),
                dah_count=int(raw_entry.get("dah_count", rhythm["dah_count"])),
                transitions=int(raw_entry.get("transitions", rhythm["transitions"])),
                repeat_pressure=str(raw_entry.get("repeat_pressure", rhythm["repeat_pressure"])),
                rhythm_diversity=float(raw_entry.get("rhythm_diversity", rhythm["rhythm_diversity"])),
            )
        )
    return entries


def read_lexicon_asset(path: Path) -> list[LexiconWord]:
    """Read an existing context lexicon JSON asset from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Lexicon JSON must contain a top-level object.")
    return lexicon_entries_from_json(data)


def matching_words(
    entries: Iterable[LexiconWord],
    *,
    known_letters: str | None = None,
    focus_letters: str | None = None,
    contains_letters: str | None = None,
    require_all_contains: bool = False,
    tag: str | None = None,
) -> list[LexiconWord]:
    """Match entries by eligibility, focus, containment, and optional tag.

    ``known_letters`` is an exclusion filter: every letter in the word must be known.
    ``focus_letters`` and ``contains_letters`` are inclusion filters: a word must contain
    at least one requested letter unless ``require_all_contains`` is set for
    ``contains_letters``.
    """

    known = normalise_allowed_letters(known_letters)
    focus = normalise_allowed_letters(focus_letters)
    contains = normalise_allowed_letters(contains_letters)

    selected: list[LexiconWord] = []
    for entry in entries:
        entry_letters = set(entry.letters)
        if known is not None and not entry_letters.issubset(known):
            continue
        if focus is not None and entry_letters.isdisjoint(focus):
            continue
        if contains is not None:
            if require_all_contains:
                if not contains.issubset(entry_letters):
                    continue
            elif entry_letters.isdisjoint(contains):
                continue
        if tag is not None and tag not in entry.tags:
            continue
        selected.append(entry)
    return selected


def selectable_words(
    entries: Iterable[LexiconWord],
    *,
    known_letters: str,
    focus_letters: str | None = None,
    tag: str | None = None,
) -> list[LexiconWord]:
    """Select entries that can be copied with known characters and optional focus letters."""

    return matching_words(
        entries,
        known_letters=known_letters,
        focus_letters=focus_letters,
        tag=tag,
    )


def audit_domains(
    entries: Iterable[LexiconWord],
    *,
    domains: Iterable[DomainDefinition] = DEFAULT_DOMAINS,
    early_sequence: str = DEFAULT_EARLY_KOCH_SEQUENCE[:12],
) -> list[DomainAuditRow]:
    """Summarise each domain's suitability for short-word context streams."""

    entry_list = list(entries)
    early = normalise_allowed_letters(early_sequence) or frozenset()
    rows: list[DomainAuditRow] = []

    for domain in domains:
        tagged = [entry for entry in entry_list if domain.key in entry.tags]
        length_counts = Counter(entry.length for entry in tagged)
        common_count = sum(1 for entry in tagged if entry.commonness in {"common", "familiar"})
        early_yield = sum(1 for entry in tagged if set(entry.letters).issubset(early))
        scores = [entry.domain_scores[domain.key] for entry in tagged]
        total = len(tagged)
        common_ratio = common_count / total if total else 0.0
        average_score = mean(scores) if scores else 0.0
        verdict = domain_verdict(total, common_ratio, early_yield, average_score)
        rows.append(
            DomainAuditRow(
                domain=domain.key,
                label=domain.label,
                total_words=total,
                three_letter=length_counts[3],
                four_letter=length_counts[4],
                five_letter=length_counts[5],
                common_ratio=round(common_ratio, 3),
                early_character_yield=early_yield,
                average_domain_score=round(average_score, 3),
                verdict=verdict,
            )
        )

    return sorted(rows, key=lambda row: (row.verdict != "keep", -row.total_words, row.domain))


def domain_verdict(total: int, common_ratio: float, early_yield: int, average_score: float) -> str:
    """Return a simple keep/maybe/reject recommendation for a domain."""

    if total >= 35 and common_ratio >= 0.25 and early_yield >= 5 and average_score >= 0.55:
        return "keep"
    if total >= 15 and common_ratio >= 0.15 and average_score >= 0.45:
        return "maybe"
    return "reject"


def lexicon_to_json(
    entries: Iterable[LexiconWord],
    *,
    description: str = "Short-word context-stream lexicon for Morse copy practice.",
) -> dict[str, object]:
    """Convert entries to a stable JSON-serialisable dictionary."""

    entry_list = list(entries)
    return {
        "schema_version": 2,
        "description": description,
        "selection_rule": "word.letters must be a subset of the learner's known characters when known letters are supplied; focus and contains letters are inclusion filters.",
        "rhythm_rule": "Morse rhythm metadata uses D for dah, I for dit, underscores for letter breaks, and rhythm_diversity from 0.0 to 1.0 for ranking early imprint opportunities.",
        "words": [asdict(entry) for entry in entry_list],
    }


def audit_to_json(rows: Iterable[DomainAuditRow]) -> dict[str, object]:
    """Convert audit rows to JSON-serialisable output."""

    return {
        "schema_version": 1,
        "description": "Domain fitness audit for short-word Morse context streams.",
        "domains": [asdict(row) for row in rows],
    }


def write_json_asset(data: Mapping[str, object], path: str | Path) -> Path:
    """Write a JSON asset with deterministic formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_audit_markdown(rows: Iterable[DomainAuditRow], path: str | Path) -> Path:
    """Write a compact Markdown table for human review."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Domain Fitness Audit",
        "",
        "This audit estimates which domains produce useful 3-, 4-, and 5-letter context-stream word pools after commonness and early-character checks.",
        "",
        "| Domain | 3 | 4 | 5 | Total | Common Ratio | Early Yield | Avg Score | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.three_letter} | {row.four_letter} | {row.five_letter} | "
            f"{row.total_words} | {row.common_ratio:.3f} | {row.early_character_yield} | "
            f"{row.average_domain_score:.3f} | {row.verdict} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
