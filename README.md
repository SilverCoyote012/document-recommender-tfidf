# Activity 3: Top-K Document Recommender with Cosine Similarity (TF-IDF)

Sistemas y Tecnologías de Administración del Conocimiento — Universidad Tecmilenio

Team: Ricardo Rubén Amaro Vázquez (AL07003220), Jorge Eliud Marroquín López (AL03011384), Juan Humberto Espinoza Arciva (AL03107716), Edson Alfonso González Vela (AL02891583).

## Objective

Build a content-based document recommender for an internal knowledge repository. Given a seed document ID or a short free-text query, the system returns the **Top-5 most similar documents** with their similarity scores, so employees can find related policies, past incidents and FAQs without searching manually. The approach is **Path A: TF-IDF + cosine similarity**, chosen for its simplicity and transparency.

## Folder contents

| Item | Description |
|---|---|
| `recommender.py` | Core logic: loading and validating the corpus, text cleaning, TF-IDF vectorization, cosine similarity, Top-K recommendation |
| `main.py` | Console interface: interactive menu or command-line arguments (`--id`, `--text`) |
| `run_tests.py` | Runs the three required tests (plus one extra) and saves the evidence |
| `requirements.txt` | Python dependencies |
| `corpus.csv` | Dataset: 40 documents with columns `id`, `title`, `text`, `category` |
| `docs_txt/` | The same 40 documents as individual `.txt` files |
| `evidencias/` | Evidence of the tests (`.txt` and `.png` for each test) |

### Expected file structure

```text
document-recommender-tfidf/   (or Actividad-3/ in the .zip)
├── recommender.py          # core logic (functions + TfidfRecommender class)
├── main.py                 # execution interface (menu / --id / --text)
├── run_tests.py            # regenerates the evidence of the tests
├── requirements.txt
├── corpus.csv              # dataset index: id, title, text, category
├── docs_txt/               # DOC001.txt ... DOC040.txt
├── evidencias/             # test1_*, test2_*, test3_*, extra_* (.txt + .png)
├── README.md
└── <...> - Actividad 3 - Reporte.pdf
```

`main.py` and `run_tests.py` look for `corpus.csv` in the same folder as the scripts, so the program runs from any working directory without changing paths.

### Dataset

The corpus has 5 categories with 8 documents each: Policy/Process, Support/Incidents, HR, IT Security and FAQ. It mixes simulated internal documents (policies, meeting minutes, tickets, FAQs, postmortems) with news- and post-style texts. Every `id` is unique (DOC001 to DOC040) and every document has a title and text (43 words on average; 36 of the 40 documents have 34 to 58 words). Four documents are intentionally short or vague (DOC016 "Ticket: Not Working" with 9 words, DOC040 with 11, DOC008 with 12 and DOC024 with 20) to observe the model's behavior with low-information documents (see `evidencias/extra_short_document`).

## Dependencies

- Python 3.9 or newer
- pandas
- scikit-learn
- pillow (only needed by `run_tests.py` to draw the PNG evidence)

```bash
pip install -r requirements.txt
```

## Execution steps

1. Open a terminal in this folder.
2. (Optional) Create a virtual environment: `python -m venv .venv`, then activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux).
3. Install the dependencies: `pip install -r requirements.txt`.
4. Run the program in one of these ways:
   - Interactive menu: `python main.py`
   - Recommend from a document ID: `python main.py --id DOC001`
   - Recommend from a free text: `python main.py --text "your text here"`
   - Optional: `--k 10` to change the number of results, `--corpus path/to/corpus.csv` to use another corpus.
5. To regenerate the evidence of the tests: `python run_tests.py` (writes to `evidencias/`).

## How it works

1. **Cleaning:** lowercase, accent removal, removal of special characters and English stopwords. The title is joined to the text so short documents keep their main topic.
2. **Vectorization:** `TfidfVectorizer` with unigrams and bigrams and sublinear term frequency.
3. **Similarity:** cosine similarity between the seed (document vector or query vector) and every document.
4. **Recommendation:** the 5 highest scores are returned. When a seed document is used, it is excluded from its own results.

## Usage example

```text
$ python main.py --id DOC009

INPUT  > Seed document: DOC009 | Ticket: VPN Will Not Connect [Support/Incidents]
OUTPUT > Top-5 similar documents (seed excluded):

#  ID      Score   Category           Title
------------------------------------------------------------------------------
1  DOC015  0.2120  Support/Incidents  Ticket: Shared Folder Access Error
2  DOC035  0.1965  FAQ                FAQ: How Do I Connect to the VPN?
3  DOC012  0.1834  Support/Incidents  Ticket: Email Password Reset
4  DOC014  0.1045  Support/Incidents  Ticket: Laptop Will Not Turn On
5  DOC010  0.0821  Support/Incidents  Ticket: Third Floor Printer Not Printing
```

Interactive mode:

```text
$ python main.py

=== Document Recommender (TF-IDF) ===
1) Recommend from a document ID
2) Recommend from a free-text query
3) List documents
0) Exit
Choose an option: 2
Paste a short text: I received a suspicious email asking me to confirm my password
```

## Evidence of the tests (`evidencias/`)

| Test | Input | File |
|---|---|---|
| Test 1: seed from Policy/Process | `DOC001` Travel Expense Approval Policy | `test1_policy_process` |
| Test 2: seed from Support/Incidents | `DOC009` Ticket: VPN Will Not Connect | `test2_support_incidents` |
| Test 3: free-text query | "I received a suspicious email asking me to confirm my password, what should I do?" | `test3_free_text` |
| Extra: limitation with a short document | `DOC016` Ticket: Not Working | `extra_short_document` |

Each file shows the input, the Top-5 with similarity scores and a brief comment on whether the recommendations are coherent and why.

## Error handling

- Unknown ID: `Error: 'Document ID not found: DOC999'` (exit code 1).
- Empty query text: `Error: The query text is empty after cleaning.`
- Missing corpus file or missing columns: descriptive error message.
- IDs are not case-sensitive (`doc001` works).
