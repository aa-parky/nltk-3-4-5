"""Command-line interface for word-list and context-stream lexicon generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .generator import DEFAULT_LENGTHS, generate_word_lists, write_word_lists
from .lexicon import (
    DEFAULT_DOMAINS,
    audit_domains,
    audit_to_json,
    build_domain_lexicon,
    lexicon_to_json,
    selectable_words,
    write_audit_markdown,
    write_json_asset,
)

LEXICON_COMMANDS = {"build-lexicon", "audit-domains", "select"}


def add_word_list_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the original plain word-list generator arguments."""

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


def build_word_parser() -> argparse.ArgumentParser:
    """Build and return the backward-compatible word-list parser."""

    parser = argparse.ArgumentParser(
        prog="nltk-3-4-5",
        description="Generate separate 3-, 4-, and 5-letter English word lists using NLTK.",
    )
    add_word_list_arguments(parser)
    return parser


def build_lexicon_parser() -> argparse.ArgumentParser:
    """Build and return the parser for JSON lexicon and domain audit commands."""

    parser = argparse.ArgumentParser(
        prog="nltk-3-4-5",
        description="Generate short-word context-stream lexicon assets for Morse copy practice.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-lexicon",
        help="Build a JSON lexicon containing short words, tags, frequency, and character metadata.",
    )
    build_parser.add_argument(
        "--output",
        default=Path("output/context_lexicon.json"),
        type=Path,
        help="JSON lexicon output path. Defaults to output/context_lexicon.json.",
    )
    build_parser.add_argument(
        "--lengths",
        nargs="+",
        default=DEFAULT_LENGTHS,
        type=int,
        help="Exact word lengths to include. Defaults to 3 4 5.",
    )

    audit_parser = subparsers.add_parser(
        "audit-domains",
        help="Build the lexicon and write domain-fitness audit outputs.",
    )
    audit_parser.add_argument(
        "--output-json",
        default=Path("output/domain_audit.json"),
        type=Path,
        help="Domain audit JSON output path. Defaults to output/domain_audit.json.",
    )
    audit_parser.add_argument(
        "--output-md",
        default=Path("output/domain_audit.md"),
        type=Path,
        help="Human-readable audit Markdown output path. Defaults to output/domain_audit.md.",
    )
    audit_parser.add_argument(
        "--early-sequence",
        default="kmuresnaptlw",
        help="Known-character sequence used to estimate early-character domain yield.",
    )

    select_parser = subparsers.add_parser(
        "select",
        help="Preview words selectable for a known character set, optional focus letters, and optional tag.",
    )
    select_parser.add_argument(
        "--known",
        required=True,
        help="Letters the learner already knows.",
    )
    select_parser.add_argument(
        "--focus",
        help="Optional focus letters that selected words should contain.",
    )
    select_parser.add_argument(
        "--tag",
        choices=sorted(domain.key for domain in DEFAULT_DOMAINS),
        help="Optional domain tag to select from.",
    )
    select_parser.add_argument(
        "--limit",
        default=50,
        type=int,
        help="Maximum number of selected words to print. Defaults to 50.",
    )
    return parser


def run_word_generation(argv: Sequence[str] | None = None) -> int:
    """Run the original 3-, 4-, and 5-letter text-file generator."""

    parser = build_word_parser()
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


def run_lexicon_command(argv: Sequence[str]) -> int:
    """Run lexicon build, audit, or selection commands."""

    parser = build_lexicon_parser()
    args = parser.parse_args(argv)

    if args.command == "build-lexicon":
        entries = build_domain_lexicon(lengths=args.lengths)
        output_path = write_json_asset(lexicon_to_json(entries), args.output)
        print(f"Lexicon: {len(entries):,} words written to {output_path}")
        return 0

    if args.command == "audit-domains":
        entries = build_domain_lexicon()
        rows = audit_domains(entries, early_sequence=args.early_sequence)
        json_path = write_json_asset(audit_to_json(rows), args.output_json)
        markdown_path = write_audit_markdown(rows, args.output_md)
        print(f"Domain audit: {len(rows):,} domains written to {json_path} and {markdown_path}")
        return 0

    if args.command == "select":
        entries = build_domain_lexicon()
        selected = selectable_words(
            entries,
            known_letters=args.known,
            focus_letters=args.focus,
            tag=args.tag,
        )[: args.limit]
        for entry in selected:
            print(entry.word)
        print(f"Selected {len(selected):,} preview words")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""

    if argv is None:
        import sys

        argv = sys.argv[1:]

    if argv and argv[0] in LEXICON_COMMANDS:
        return run_lexicon_command(argv)
    return run_word_generation(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
