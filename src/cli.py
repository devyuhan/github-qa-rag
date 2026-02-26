"""CLI entry point — ingest and interactive Q&A."""

import argparse
import sys

from src.config import Settings


def _cmd_ingest(args):
    from src.ingest import ingest

    extensions = args.extensions or list(Settings.default_extensions)
    ingest(
        owner=args.owner,
        repo=args.repo,
        branch=args.branch,
        extensions=extensions,
    )


def _cmd_ask(args):
    from src.query import ask

    eval_mode = args.eval

    if eval_mode:
        from src.evaluate import evaluate

    print("Interactive Q&A  (type 'quit' or 'exit' to leave)\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in ("quit", "exit"):
            break

        response = ask(question)

        print(f"\nAnswer:\n{response}\n")

        # Show source files
        if response.source_nodes:
            print("Sources:")
            for node in response.source_nodes:
                path = node.metadata.get("file_path", "unknown")
                score = f"{node.score:.3f}" if node.score is not None else "n/a"
                print(f"  - {path}  (score: {score})")
            print()

        if eval_mode:
            print("Evaluating ...")
            results = evaluate(question, response)
            for metric, data in results.items():
                status = "PASS" if data["passing"] else "FAIL"
                print(f"  {metric}: {status}  (score: {data['score']})")
                if data["feedback"]:
                    print(f"    feedback: {data['feedback']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        prog="github-qa-rag",
        description="RAG-based Q&A over GitHub repositories",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    ingest_p = subparsers.add_parser("ingest", help="Ingest a GitHub repository")
    ingest_p.add_argument("--owner", required=True, help="GitHub owner/org")
    ingest_p.add_argument("--repo", required=True, help="Repository name")
    ingest_p.add_argument("--branch", default="main", help="Branch (default: main)")
    ingest_p.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="File extensions to include (e.g. .py .md)",
    )

    # --- ask ---
    ask_p = subparsers.add_parser("ask", help="Interactive Q&A")
    ask_p.add_argument(
        "--eval", action="store_true", help="Run faithfulness/relevancy evaluation"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "ask":
        _cmd_ask(args)


if __name__ == "__main__":
    main()
