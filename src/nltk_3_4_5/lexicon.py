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
            LexiconWord(
                word=word,
                length=len(word),
                letters=tuple(sorted(set(word))),
                tags=tuple(sorted(domain_scores)),
                frequency=frequency,
                frequency_rank=rank,
                commonness=commonness_for_rank(rank, frequency),
                domain_scores=dict(sorted(domain_scores.items())),
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
            )
        )
    return entries


def read_lexicon_asset(path: Path) -> list[LexiconWord]:
    """Read an existing context lexicon JSON asset from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Lexicon JSON must contain a top-level object.")
    return lexicon_entries_from_json(data)


def selectable_words(
    entries: Iterable[LexiconWord],
    *,
    known_letters: str,
    focus_letters: str | None = None,
    tag: str | None = None,
) -> list[LexiconWord]:
    """Select entries that can be copied with known characters and optional focus letters."""

    known = normalise_allowed_letters(known_letters) or frozenset()
    focus = normalise_allowed_letters(focus_letters)

    selected: list[LexiconWord] = []
    for entry in entries:
        entry_letters = set(entry.letters)
        if not entry_letters.issubset(known):
            continue
        if focus is not None and entry_letters.isdisjoint(focus):
            continue
        if tag is not None and tag not in entry.tags:
            continue
        selected.append(entry)
    return selected


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


def lexicon_to_json(entries: Iterable[LexiconWord]) -> dict[str, object]:
    """Convert entries to a stable JSON-serialisable dictionary."""

    entry_list = list(entries)
    return {
        "schema_version": 1,
        "description": "Short-word context-stream lexicon for Morse copy practice.",
        "selection_rule": "word.letters must be a subset of the learner's known characters; optional focus letters should intersect word.letters.",
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
