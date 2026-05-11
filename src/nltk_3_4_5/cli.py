"""Command-line interface for the word-list generator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .generator import DEFAULT_LENGTHS, generate_word_lists, write_word_lists


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="nltk-3-4-5",
        description="Generate separate 3-, 4-, and 5-letter English word lists using NLTK.",
    )
    parser.add_argument(
        "--letters",
        metavar="LETTERS",
        help=(
            "Optional allowed alphabet. When supplied, only words composed entirely "
            "from these letters are included. For example: --letters etianmsurwdkgo"
        ),
    )
    parser.add_argument(
        "--include-proper-nouns",
        action="store_true",
        help="Include capitalised corpus entries. By default, likely proper nouns are excluded.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        type=Path,
        help="Directory where generated text files will be written. Defaults to ./output.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=DEFAULT_LENGTHS,
        type=int,
        help="Exact word lengths to generate. Defaults to 3 4 5.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""

    parser = build_parser()
    args = parser.parse_args(argv)

    word_lists = generate_word_lists(
        lengths=args.lengths,
        allowed_letters=args.letters,
        include_proper_nouns=args.include_proper_nouns,
    )
    written_files = write_word_lists(word_lists, args.output_dir)

    for length in sorted(written_files):
        print(
            f"{length}-letter words: {len(word_lists[length]):,} "
            f"written to {written_files[length]}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
