import argparse

from trainers.pushforward import PFTrainer
from trainers.simple import SimpleTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the BubbleML dataset')
    parser.add_argument("--conf", type=str, default="conf/example.yaml")
    parser.add_argument("--trainer", type=str, default="PF")
    args = parser.parse_args()
    if args.trainer == "simple":
        trainer = SimpleTrainer(args.conf)
    elif args.trainer == "PF":
        trainer = PFTrainer(args.conf)
    else:
        raise ValueError("Trainer not set")
    trainer.train()