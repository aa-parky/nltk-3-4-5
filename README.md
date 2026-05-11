# nltk-3-4-5

`nltk-3-4-5` is a small Python command-line application that uses the **NLTK `words` corpus** to generate three separate English word lists: one for **3-letter words**, one for **4-letter words**, and one for **5-letter words**. It is configured for local development with **pyenv set to Python 3.13**.

The application writes newline-delimited `.txt` files into an output directory. By default it excludes capitalised entries, which helps avoid likely proper nouns, and it keeps only alphabetic words.

## Project structure

| Path | Purpose |
|---|---|
| `.python-version` | Sets the local pyenv version to Python 3.13. |
| `pyproject.toml` | Declares package metadata, dependencies, console script, and test configuration. |
| `src/nltk_3_4_5/generator.py` | Contains the NLTK corpus loading, filtering, generation, and file-writing logic. |
| `src/nltk_3_4_5/cli.py` | Provides the `nltk-3-4-5` command-line interface. |
| `tests/test_generator.py` | Covers filtering, grouping, sorting, and output writing. |
| `output/` | Default folder for generated word-list files. |

## Setup

From a local checkout of the repository, run:

```bash
pyenv local 3.13
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first generation run downloads the small NLTK `words` corpus if it is not already available in your local NLTK data directory.

## Generate the default lists

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

## Additional options

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

The NLTK `words` corpus is a practical default for this project because it behaves like a broad English word list rather than a sentence corpus. The app normalises accepted words to lowercase, deduplicates them, sorts them alphabetically, and writes one file per requested word length.
