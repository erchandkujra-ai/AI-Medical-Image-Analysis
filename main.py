
import argparse

from src.train import main as train_model
from src.evaluate import main as evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="AI Medical Image Analysis prototype"
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate"],
        help="Choose training or independent test evaluation"
    )
    args = parser.parse_args()

    if args.action == "train":
        train_model()
    elif args.action == "evaluate":
        evaluate_model()


if __name__ == "__main__":
    main()
