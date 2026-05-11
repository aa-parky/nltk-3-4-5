from __future__ import annotations

import json

from nltk_3_4_5.cli import main
from nltk_3_4_5.lexicon import (
    DomainDefinition,
    LexiconWord,
    audit_domains,
    audit_to_json,
    build_foundation_lexicon,
    commonness_for_rank,
    lexicon_entries_from_json,
    lexicon_to_json,
    lexicon_word,
    matching_words,
    read_lexicon_asset,
    selectable_words,
    write_json_asset,
)


def test_commonness_for_rank_groups_frequency_usefully() -> None:
    assert commonness_for_rank(1, 50) == "common"
    assert commonness_for_rank(8_000, 10) == "familiar"
    assert commonness_for_rank(20_000, 1) == "rare"
    assert commonness_for_rank(None, 0) == "unranked"


def test_matching_words_distinguishes_known_from_contains() -> None:
    entries = [
        LexiconWord(
            word="mum",
            length=3,
            letters=("m", "u"),
            tags=("foundation",),
            frequency=2,
            frequency_rank=100,
            commonness="common",
            domain_scores={"foundation": 1.0},
        ),
        LexiconWord(
            word="kite",
            length=4,
            letters=("e", "i", "k", "t"),
            tags=("foundation",),
            frequency=2,
            frequency_rank=200,
            commonness="common",
            domain_scores={"foundation": 1.0},
        ),
        LexiconWord(
            word="rum",
            length=3,
            letters=("m", "r", "u"),
            tags=("foundation",),
            frequency=1,
            frequency_rank=300,
            commonness="common",
            domain_scores={"foundation": 1.0},
        ),
    ]

    assert matching_words(entries, known_letters="km") == []
    assert [entry.word for entry in matching_words(entries, contains_letters="km")] == ["mum", "kite", "rum"]
    assert [entry.word for entry in matching_words(entries, contains_letters="km", require_all_contains=True)] == []


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


def test_build_foundation_lexicon_keeps_broad_brown_attested_words() -> None:
    entries = build_foundation_lexicon(
        lengths=(3, 4),
        corpus_words=("mum", "rum", "emu", "mars", "Moon", "rare"),
        frequency_counts={"mum": 2, "rum": 1, "emu": 0, "mars": 3, "rare": 0},
        frequency_ranks={"mum": 1_000, "rum": 8_000, "mars": 20_000},
    )

    assert [entry.word for entry in entries] == ["mars", "mum", "rum"]
    assert {entry.tags for entry in entries} == {("foundation",)}
    assert entries[0].domain_scores == {"foundation": 1.0}
    assert entries[0].commonness == "rare"


def test_lexicon_json_and_audit_json_are_serialisable(tmp_path) -> None:
    entry = lexicon_word(
        word="ship",
        tags=("maritime",),
        frequency=12,
        frequency_rank=500,
        commonness="common",
        domain_scores={"maritime": 1.0},
    )
    lexicon = lexicon_to_json([entry])
    path = write_json_asset(lexicon, tmp_path / "lexicon.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 2
    assert loaded["words"][0]["word"] == "ship"
    assert loaded["words"][0]["morse"] == "... .... .. .--."
    assert loaded["words"][0]["rhythm_signature"] == "III_IIII_II_IDDI"
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

    assert len(entries) == 1
    assert entries[0].word == "cat"
    assert entries[0].letters == ("a", "c", "t")
    assert entries[0].tags == ("animals",)
    assert entries[0].frequency == 4
    assert entries[0].frequency_rank == 200
    assert entries[0].commonness == "common"
    assert entries[0].domain_scores == {"animals": 1.0}
    assert entries[0].morse == "-.-. .- -"
    assert entries[0].rhythm_signature == "DIDI_ID_D"
    assert entries[0].rhythm_diversity > 0


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


def test_build_foundation_lexicon_command_writes_json_asset(tmp_path, capsys) -> None:
    output_path = tmp_path / "foundation_lexicon.json"

    exit_code = main(["build-foundation-lexicon", "--output", str(output_path), "--lengths", "3"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Foundation lexicon:" in output
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["description"] == "Broad Brown-attested short-word lexicon for early Morse copy stages."
    assert {word["length"] for word in data["words"]} == {3}
    assert {"foundation"} <= {tag for word in data["words"] for tag in word["tags"]}


def test_count_command_allows_focus_without_known_as_containment_count(tmp_path, capsys) -> None:
    path = tmp_path / "foundation_lexicon.json"
    write_json_asset(
        lexicon_to_json(
            [
                LexiconWord(
                    word="mum",
                    length=3,
                    letters=("m", "u"),
                    tags=("foundation",),
                    frequency=2,
                    frequency_rank=100,
                    commonness="common",
                    domain_scores={"foundation": 1.0},
                ),
                LexiconWord(
                    word="era",
                    length=3,
                    letters=("a", "e", "r"),
                    tags=("foundation",),
                    frequency=3,
                    frequency_rank=200,
                    commonness="common",
                    domain_scores={"foundation": 1.0},
                ),
            ]
        ),
        path,
    )

    exit_code = main(["count", "--focus", "km", "--lexicon", str(path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Known letters: not restricted" in output
    assert "Focus letters: km" in output
    assert "Selectable words: 1" in output
    assert "foundation: 1" in output


def test_count_command_supports_explicit_contains_any_and_all(tmp_path, capsys) -> None:
    path = tmp_path / "foundation_lexicon.json"
    write_json_asset(
        lexicon_to_json(
            [
                LexiconWord(
                    word="make",
                    length=4,
                    letters=("a", "e", "k", "m"),
                    tags=("foundation",),
                    frequency=5,
                    frequency_rank=100,
                    commonness="common",
                    domain_scores={"foundation": 1.0},
                ),
                LexiconWord(
                    word="mum",
                    length=3,
                    letters=("m", "u"),
                    tags=("foundation",),
                    frequency=2,
                    frequency_rank=200,
                    commonness="common",
                    domain_scores={"foundation": 1.0},
                ),
            ]
        ),
        path,
    )

    any_exit_code = main(["count", "--contains", "km", "--lexicon", str(path)])
    any_output = capsys.readouterr().out
    all_exit_code = main(["count", "--contains", "km", "--contains-all", "--lexicon", str(path)])
    all_output = capsys.readouterr().out

    assert any_exit_code == 0
    assert "Selectable words: 2" in any_output
    assert all_exit_code == 0
    assert "Contains letters: km (all)" in all_output
    assert "Selectable words: 1" in all_output


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


def test_rhythm_metadata_for_generated_lexicon_word() -> None:
    entry = lexicon_word(
        word="mud",
        tags=("nature",),
        frequency=4,
        frequency_rank=400,
        commonness="common",
        domain_scores={"nature": 1.0},
    )

    assert entry.morse == "-- ..- -.."
    assert entry.rhythm_signature == "DD_IID_DII"
    assert entry.dit_count == 4
    assert entry.dah_count == 4
    assert entry.transitions == 3
    assert entry.repeat_pressure == "medium"
    assert 0.0 < entry.rhythm_diversity <= 1.0


def test_sample_command_prefers_rhythmic_diverse_words(tmp_path, capsys) -> None:
    path = tmp_path / "context_lexicon.json"
    write_json_asset(
        lexicon_to_json(
            [
                lexicon_word(
                    word="mum",
                    tags=("home",),
                    frequency=2,
                    frequency_rank=200,
                    commonness="common",
                    domain_scores={"home": 1.0},
                ),
                lexicon_word(
                    word="mud",
                    tags=("nature",),
                    frequency=4,
                    frequency_rank=400,
                    commonness="common",
                    domain_scores={"nature": 1.0},
                ),
                lexicon_word(
                    word="era",
                    tags=("work",),
                    frequency=3,
                    frequency_rank=300,
                    commonness="common",
                    domain_scores={"work": 1.0},
                ),
            ]
        ),
        path,
    )

    exit_code = main(
        [
            "sample",
            "--focus",
            "mu",
            "--prefer",
            "rhythmic-diverse",
            "--limit",
            "2",
            "--lexicon",
            str(path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Matched words: 2" in output
    assert "Sample preference: rhythmic-diverse" in output
    assert "Sample words: 2" in output
    assert "mud: -- ..- -.. | DD_IID_DII" in output
    assert "rhythm=" in output


def test_sample_command_balances_multiple_focus_letters_when_rhythmic(tmp_path, capsys) -> None:
    path = tmp_path / "context_lexicon.json"
    write_json_asset(
        lexicon_to_json(
            [
                lexicon_word(
                    word="ark",
                    tags=("travel",),
                    frequency=5,
                    frequency_rank=100,
                    commonness="common",
                    domain_scores={"travel": 1.0},
                ),
                lexicon_word(
                    word="neck",
                    tags=("body",),
                    frequency=5,
                    frequency_rank=110,
                    commonness="common",
                    domain_scores={"body": 1.0},
                ),
                lexicon_word(
                    word="tick",
                    tags=("nature",),
                    frequency=5,
                    frequency_rank=120,
                    commonness="common",
                    domain_scores={"nature": 1.0},
                ),
                lexicon_word(
                    word="tuck",
                    tags=("home",),
                    frequency=4,
                    frequency_rank=200,
                    commonness="common",
                    domain_scores={"home": 1.0},
                ),
                lexicon_word(
                    word="emu",
                    tags=("animals",),
                    frequency=3,
                    frequency_rank=300,
                    commonness="common",
                    domain_scores={"animals": 1.0},
                ),
                lexicon_word(
                    word="milk",
                    tags=("food",),
                    frequency=3,
                    frequency_rank=400,
                    commonness="common",
                    domain_scores={"food": 1.0},
                ),
            ]
        ),
        path,
    )

    exit_code = main(
        [
            "sample",
            "--focus",
            "kmu",
            "--prefer",
            "rhythmic-diverse",
            "--limit",
            "3",
            "--lexicon",
            str(path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Sample words: 3" in output
    assert "tuck:" in output
    assert "emu:" in output
    assert "milk:" in output
    assert "ark:" not in output
