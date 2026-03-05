import argparse
from src.core.config import load_config
from src.core.rag_system import RAGSystem
from src.evaluation.evaluator import RAGEvaluator


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = load_config()
    rag = RAGSystem(config)

    if args.eval:
        evaluator = RAGEvaluator(rag)
        metrics = evaluator.evaluate("data/eval/eval_set.json")
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        return

    while True:
        query = input("Ask (or type exit): ")

        if query.lower() == "exit":
            break

        answer = rag.ask(query)

        print("\n=== Answer ===\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()