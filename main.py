"""Console interface for the TF-IDF document recommender.

Interactive menu:
    python main.py

Direct (non-interactive) use:
    python main.py --id DOC001
    python main.py --text "my laptop cannot connect to the vpn"
"""

from __future__ import annotations

import argparse
import sys

from recommender import DEFAULT_CORPUS, DEFAULT_TOP_K, TfidfRecommender, format_results, load_corpus


def run_by_id(rec: TfidfRecommender, doc_id: str, k: int) -> None:
    seed = rec.get_document(doc_id)
    results = rec.recommend_by_id(doc_id, k)
    print(f"\nINPUT  > Seed document: {seed['id']} | {seed['title']} [{seed['category']}]")
    print(f"OUTPUT > Top-{k} similar documents (seed excluded):\n")
    print(format_results(results))


def run_by_text(rec: TfidfRecommender, query: str, k: int) -> None:
    results = rec.recommend_by_text(query, k)
    print(f'\nINPUT  > Query text: "{query}"')
    print(f"OUTPUT > Top-{k} most similar documents:\n")
    print(format_results(results))


def interactive(rec: TfidfRecommender, k: int) -> None:
    menu = (
        "\n=== Document Recommender (TF-IDF) ===\n"
        "1) Recommend from a document ID\n"
        "2) Recommend from a free-text query\n"
        "3) List documents\n"
        "0) Exit\n"
        "Choose an option: "
    )
    while True:
        choice = input(menu).strip()
        try:
            if choice == "1":
                run_by_id(rec, input("Document ID (e.g., DOC001): "), k)
            elif choice == "2":
                run_by_text(rec, input("Paste a short text: "), k)
            elif choice == "3":
                print(rec.corpus[["id", "title", "category"]].to_string(index=False))
            elif choice == "0":
                print("Goodbye.")
                return
            else:
                print("Invalid option, try again.")
        except (KeyError, ValueError) as err:
            print(f"Error: {err}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF document recommender")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--id", help="Seed document ID (e.g., DOC001)")
    group.add_argument("--text", help="Free-text query")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Path to corpus.csv")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of recommendations")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rec = TfidfRecommender(load_corpus(args.corpus))
        if args.id:
            run_by_id(rec, args.id, args.k)
        elif args.text:
            run_by_text(rec, args.text, args.k)
        else:
            interactive(rec, args.k)
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
