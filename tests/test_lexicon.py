from __future__ import annotations

import json

from nltk_3_4_5.cli import main
from nltk_3_4_5.lexicon import (
    DomainDefinition,
    LexiconWord,
    audit_domains,
    audit_to_json,
    commonness_for_rank,
    lexicon_entries_from_json,
    lexicon_to_json,
    read_lexicon_asset,
    selectable_words,
    write_json_asset,
)


def test_commonness_for_rank_groups_frequency_usefully() -> None:
    assert commonness_for_rank(1, 50) == "common"
    assert commonness_for_rank(8_000, 10) == "familiar"
    assert commonness_for_rank(20_000, 1) == "rare"
    assert commonness_for_rank(None, 0) == "unranked"


def test_selectable_words_filters_by_known_focus_and_tag() -> None:
    entries = [
        LexiconWord(
            word="ham",
            length=3,
            letters=("a", "h", "m"),
            tags=("food",),
            frequency=7,
            frequency_rank=100,
            commonness="common",
            domain_scores={"food": 1.0},
        ),
        LexiconWord(
            word="cat",
            length=3,
            letters=("a", "c", "t"),
            tags=("animals",),
            frequency=4,
            frequency_rank=200,
            commonness="common",
            domain_scores={"animals": 1.0},
        ),
        LexiconWord(
            word="cake",
            length=4,
            letters=("a", "c", "e", "k"),
            tags=("food",),
            frequency=3,
            frequency_rank=300,
            commonness="common",
            domain_scores={"food": 1.0},
        ),
    ]

    selected = selectable_words(entries, known_letters="ahmcte", focus_letters="m", tag="food")

    assert [entry.word for entry in selected] == ["ham"]


def test_audit_domains_counts_lengths_and_verdicts() -> None:
    domain = DomainDefinition(key="food", label="Food", seeds=("food",))
    entries = [
        LexiconWord(
            word=f"aa{i}",
            length=3,
            letters=("a",),
            tags=("food",),
            frequency=10,
            frequency_rank=100 + i,
            commonness="common",
            domain_scores={"food": 0.9},
        )
        for i in range(10)
    ] + [
        LexiconWord(
            word=f"bbbb{i}",
            length=5,
            letters=("b",),
            tags=("food",),
            frequency=1,
            frequency_rank=20_000 + i,
            commonness="rare",
            domain_scores={"food": 0.7},
        )
        for i in range(30)
    ]

    rows = audit_domains(entries, domains=(domain,), early_sequence="ab")

    assert rows[0].domain == "food"
    assert rows[0].three_letter == 10
    assert rows[0].five_letter == 30
    assert rows[0].total_words == 40
    assert rows[0].verdict == "keep"


def test_lexicon_json_and_audit_json_are_serialisable(tmp_path) -> None:
    entry = LexiconWord(
        word="ship",
        length=4,
        letters=("h", "i", "p", "s"),
        tags=("maritime",),
        frequency=12,
        frequency_rank=500,
        commonness="common",
        domain_scores={"maritime": 1.0},
    )
    lexicon = lexicon_to_json([entry])
    path = write_json_asset(lexicon, tmp_path / "lexicon.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["words"][0]["word"] == "ship"
    assert audit_to_json([])["domains"] == []


def test_lexicon_entries_from_json_rebuilds_word_records() -> None:
    data = {
        "words": [
            {
                "word": "cat",
                "length": 3,
                "letters": ["a", "c", "t"],
                "tags": ["animals"],
                "frequency": 4,
                "frequency_rank": 200,
                "commonness": "common",
                "domain_scores": {"animals": 1.0},
            }
        ]
    }

    entries = lexicon_entries_from_json(data)

    assert entries == [
        LexiconWord(
            word="cat",
            length=3,
            letters=("a", "c", "t"),
            tags=("animals",),
            frequency=4,
            frequency_rank=200,
            commonness="common",
            domain_scores={"animals": 1.0},
        )
    ]


def test_read_lexicon_asset_loads_context_lexicon_json(tmp_path) -> None:
    path = tmp_path / "context_lexicon.json"
    path.write_text(
        json.dumps(
            {
                "description": "Test lexicon",
                "schema_version": 1,
                "words": [
                    {
                        "word": "ham",
                        "length": 3,
                        "letters": ["a", "h", "m"],
                        "tags": ["food"],
                        "frequency": 7,
                        "frequency_rank": None,
                        "commonness": "unranked",
                        "domain_scores": {"food": 0.8},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    entries = read_lexicon_asset(path)

    assert [entry.word for entry in entries] == ["ham"]
    assert entries[0].frequency_rank is None


def test_count_command_reports_total_lengths_and_tags(tmp_path, capsys) -> None:
    path = tmp_path / "context_lexicon.json"
    write_json_asset(
        lexicon_to_json(
            [
                LexiconWord(
                    word="ham",
                    length=3,
                    letters=("a", "h", "m"),
                    tags=("food",),
                    frequency=7,
                    frequency_rank=100,
                    commonness="common",
                    domain_scores={"food": 1.0},
                ),
                LexiconWord(
                    word="cake",
                    length=4,
                    letters=("a", "c", "e", "k"),
                    tags=("food",),
                    frequency=3,
                    frequency_rank=300,
                    commonness="common",
                    domain_scores={"food": 1.0},
                ),
                LexiconWord(
                    word="cat",
                    length=3,
                    letters=("a", "c", "t"),
                    tags=("animals",),
                    frequency=4,
                    frequency_rank=200,
                    commonness="common",
                    domain_scores={"animals": 1.0},
                ),
            ]
        ),
        path,
    )

    exit_code = main(["count", "--known", "ahm", "--lexicon", str(path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Selectable words: 1" in output
    assert "3-letter words: 1" in output
    assert "food: 1" in output
    assert "animals" not in output
