# nltk-3-4-5

`nltk-3-4-5` is a small Python command-line toolkit for generating short English word assets from **NLTK**. It started as a simple generator for separate **3-letter**, **4-letter**, and **5-letter** word lists, and now also includes an offline **context-stream lexicon** workflow for Morse copy practice.

The project is configured for local development with **pyenv set to Python 3.13**. It favours offline generation: the linguistic and curation work happens in this repository, while an app can consume stable JSON assets without needing to run NLTK at runtime.

## Project structure

| Path | Purpose |
|---|---|
| `.python-version` | Sets the local pyenv version to Python 3.13. |
| `pyproject.toml` | Declares package metadata, dependencies, console script, and test configuration. |
| `src/nltk_3_4_5/generator.py` | Contains the original NLTK word-list loading, filtering, generation, and file-writing logic. |
| `src/nltk_3_4_5/lexicon.py` | Builds domain-tagged, frequency-aware, JSON-ready short-word lexicons. |
| `src/nltk_3_4_5/cli.py` | Provides the `nltk-3-4-5` command-line interface. |
| `tests/` | Covers word-list generation, lexicon selection, audit scoring, and JSON writing. |
| `output/` | Default folder for generated text and JSON assets. |

## Setup

From a local checkout of the repository, run:

```bash
pyenv local 3.13
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first generation run downloads the required NLTK corpora if they are not already available in your local NLTK data directory. The plain word-list generator uses `words`; the context lexicon also uses `wordnet` and `brown`.

## Generate the default 3-, 4-, and 5-letter lists

```bash
nltk-3-4-5
```

This creates the following files:

| File | Contents |
|---|---|
| `output/3_letter_words.txt` | Sorted unique English words with exactly 3 letters. |
| `output/4_letter_words.txt` | Sorted unique English words with exactly 4 letters. |
| `output/5_letter_words.txt` | Sorted unique English words with exactly 5 letters. |

## Generate lists using only selected characters

If you want to support Morse-learning exercises or any other constrained-character practice, pass an allowed alphabet with `--letters`. Only words composed entirely from those letters are included.

```bash
nltk-3-4-5 --letters etianmsurwdkgo
```

You can also choose a different output directory:

```bash
nltk-3-4-5 --letters etianmsurwdkgo --output-dir output/morse_stage_1
```

## Build a context-stream lexicon

The context-stream workflow creates a single JSON file containing words, lengths, required letters, domain tags, Brown-corpus frequency information, broad commonness labels, and per-domain confidence scores.

```bash
nltk-3-4-5 build-lexicon
```

By default this writes:

```text
output/context_lexicon.json
```

A word record looks like this:

```json
{
  "word": "ship",
  "length": 4,
  "letters": ["h", "i", "p", "s"],
  "tags": ["maritime"],
  "frequency": 73,
  "frequency_rank": 2113,
  "commonness": "common",
  "domain_scores": {
    "maritime": 1.0
  },
  "morse": "... .... .. .--.",
  "rhythm_signature": "III_IIII_II_IDDI",
  "dit_count": 9,
  "dah_count": 2,
  "transitions": 2,
  "repeat_pressure": "high",
  "rhythm_diversity": 0.47
}
```

The rhythm fields let the app treat the word as sound before text. `morse` keeps the letter-by-letter Morse pattern, `rhythm_signature` stores the same pattern as `I` for dit and `D` for dah, `dit_count` and `dah_count` describe balance, `transitions` counts dit/dah alternations, `repeat_pressure` flags heavily repeated patterns, and `rhythm_diversity` provides a sortable score between `0.0` and `1.0` for rhythm-rich practice material.

The intended app-side selection rule is deliberately simple, but the distinction between **known** and **focus** letters matters:

> A word is selectable when its `letters` are a subset of the learner's known characters. In other words, known letters are an exclusion filter: any word containing an unknown character is rejected. For target-character practice, the selected word should also intersect the current focus letters, meaning it contains at least one of the requested focus characters.

## Build a foundation lexicon for early stages

Domain-tagged words are useful once the learner has enough characters for themed streams, but they can be too aggressive at the very beginning. For early Koch-style stages, build a broader **foundation** lexicon that keeps short NLTK dictionary words with at least one Brown-corpus occurrence and tags them as `foundation` rather than forcing them into a topic.

```bash
nltk-3-4-5 build-foundation-lexicon
```

By default this writes:

```text
output/foundation_lexicon.json
```

You can then count against the broader early-stage asset:

```bash
nltk-3-4-5 count --known kmure --lexicon output/foundation_lexicon.json
```

This gives the app two practical pools: `foundation_lexicon.json` for early character acquisition, and `context_lexicon.json` for later themed/domain practice.

| Asset | Best use | Filtering strategy |
|---|---|---|
| `output/foundation_lexicon.json` | Early stages where themed domains are too sparse. | Broad dictionary words that are also attested in Brown. |
| `output/context_lexicon.json` | Later contextual streams, themed drills, and domain-aware practice. | Domain candidates from WordNet and curated seeds, with frequency metadata. |

## Count selectable words from an existing lexicon

The `count` command reads an existing JSON lexicon asset and reports how many words are selectable for a supplied set of known characters. It does not rebuild the lexicon, so it is a quick way to check whether a learner stage has enough listening material. Use the foundation lexicon for early-stage breadth, and the context lexicon when you specifically want themed/domain material.

The `--known` option is deliberately restrictive: `--known km` means “only words made entirely from `k` and `m`”, not “words that contain `k` or `m`”. To ask the exploratory question “which words contain `k` or `m`?”, use standalone `--focus km` or the more explicit `--contains km`. To require both letters to appear in the same word, add `--contains-all`.

```bash
nltk-3-4-5 count --known kmuresnaptlw
```

The output includes the total selectable words, a breakdown by word length, and a tag breakdown. You can narrow the count with a focus-letter requirement or a single tag:

```bash
nltk-3-4-5 count --known kmuresnaptlw --focus km
nltk-3-4-5 count --known kmuresnaptlw --tag food
nltk-3-4-5 count --known kmuresnaptlw --tag foundation --lexicon output/foundation_lexicon.json
nltk-3-4-5 count --focus km --lexicon output/foundation_lexicon.json
nltk-3-4-5 count --contains km --contains-all --lexicon output/foundation_lexicon.json
```

These modes answer different development questions:

| Example | Meaning |
|---|---|
| `--known km` | Count words made only from `k` and `m`. |
| `--known kmures --focus km` | Count words made only from known letters and containing `k` or `m`. |
| `--focus km` | Count words containing `k` or `m`, with no known-letter restriction. |
| `--contains km --contains-all` | Count words containing both `k` and `m`, with no known-letter restriction. |

If the asset is stored somewhere other than `output/context_lexicon.json`, pass its path explicitly:

```bash
nltk-3-4-5 count --known kmuresnaptlw --lexicon path/to/context_lexicon.json
```

## Audit domain fitness

Not every possible theme produces good short-word listening material. Some domains have plenty of short, recognisable words; others collapse under the 3/4/5-letter and known-character constraints. The domain audit estimates which domains are worth keeping.

```bash
nltk-3-4-5 audit-domains
```

This writes two files:

| File | Purpose |
|---|---|
| `output/domain_audit.json` | Machine-readable domain-fitness summary. |
| `output/domain_audit.md` | Human-readable Markdown audit table. |

The audit includes counts by word length, the ratio of common or familiar words, early-character yield, average domain score, and a simple `keep`, `maybe`, or `reject` verdict.

## Sample rhythm-rich focus words

The `sample` command is the learner-facing complement to `count`. It returns a small candidate set from an existing lexicon, using the same focus, contains, known-letter, and tag filters, then ranks the result for teaching use.

```bash
nltk-3-4-5 sample --focus kmu --prefer rhythmic-diverse
```

The `rhythmic-diverse` preference favours words with higher `rhythm_diversity`, stronger dit/dah transitions, and richer total Morse shape. When more than one focus or contains letter is supplied, the sampler also balances exposure across those requested characters. This prevents a high-scoring character such as `k` from dominating the whole sample when the intended stream is `k`, `m`, and `u` together. Use `--limit 0` when you want the maximum matched set printed in ranked order rather than the default twelve-word sample.

| Option | Meaning |
|---|---|
| `--focus kmu` | Match words containing at least one of `k`, `m`, or `u`. |
| `--contains kmu --contains-all` | Match words containing all supplied characters. |
| `--known kmures` | Optionally restrict the sample to words made only from known letters. |
| `--prefer rhythmic-diverse` | Prefer rhythm-rich Morse shapes while balancing exposure across multiple focus or contains letters. |
| `--prefer frequency` | Rank by Brown-corpus frequency where available. |
| `--prefer word` | Rank alphabetically. |
| `--prefer random` | Draw from the matched set without rhythm or frequency bias, while still displaying rhythm metadata for review. |
| `--limit 20` | Control the number of sample words printed. |
| `--limit 0` | Print the full matched set instead of stopping at a sample size. |

For example, `mud` is exported and displayed with both its Morse and rhythm signature:

```text
mud: -- ..- -.. | DD_IID_DII | rhythm=0.72
```

This creates the missing **context stream** layer: Koch and Farnsworth still control character order and timing, while the lexicon can now choose words that provide a strong rhythmic imprint for the current focus set. In practice, `sample --focus kmu --prefer rhythmic-diverse` is not just “top twelve rhythm scores”; it tries to surface `k`, `m`, and `u` across the returned teaching set so the student receives a broader early imprint. When you need to audit the full candidate pool, run `nltk-3-4-5 sample --focus kmu --limit 0`; when you want a non-polluting draw for practice, run `nltk-3-4-5 sample --focus kmu --prefer random --limit 12`.

## Preview selectable words

You can preview words that would be selectable for a learner's known character set. This does not replace the app's runtime selection; it is a development aid for checking whether a domain has enough usable depth.

```bash
nltk-3-4-5 select --known kmuresnaptlw --tag food --limit 30
```

You can also require focus letters:

```bash
nltk-3-4-5 select --known kmuresnaptlw --focus km --tag food
```

## Context lexicon design

The context-stream lexicon is intentionally **theme-agnostic**. Domains are not chosen because they are interesting in the abstract; they are chosen because they produce useful short words after filtering. The included starter domains are candidates for audit and curation, not a final curriculum.

| Layer | Purpose |
|---|---|
| NLTK `words` | Validates alphabetic English candidates. |
| WordNet | Expands domains from seed concepts and semantic relations. |
| Curated seed words | Improves quality where WordNet is too broad, too technical, or too sparse. |
| Brown corpus frequency | Ranks common words above obscure dictionary entries. |
| Character metadata | Lets an app select words by known Morse characters without linguistic processing. |
| Domain audit | Identifies domains with enough depth, variance, and early-character survivability. |

The generated JSON is designed for the Copy philosophy: listening first, self-review after, and no need to treat missed copy as failure. Missing copy is expected; the important behaviour is continuing forward while the ear gradually recognises more of the stream.

## Additional plain word-list options

| Option | Behaviour |
|---|---|
| `--letters LETTERS` | Restricts words to those made only from the supplied alphabetic characters. Non-letters in the option value are ignored. |
| `--include-proper-nouns` | Includes capitalised corpus entries after lowercasing them. By default, capitalised entries are excluded. |
| `--output-dir PATH` | Writes generated files to `PATH` instead of `output/`. |
| `--lengths 3 4 5` | Changes the exact word lengths generated. The default remains `3 4 5`. |

## Run tests

```bash
pytest
```

## Notes on corpus choice

The original generator uses the NLTK `words` corpus because it behaves like a broad English word list rather than a sentence corpus. The context lexicon adds WordNet for domain expansion and the Brown corpus for simple frequency-based commonness. The result should still be treated as a candidate dataset: the audit helps identify strong domains, and light human curation should remove words that feel obscure, misleading, offensive, or unsuitable for a listening stream.
