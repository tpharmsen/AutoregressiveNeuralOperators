import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the BubbleML dataset')
    parser.add_argument("--conf", type=str, default="conf/example.yaml")
    parser.add_argument("--trainer", type=str, default="PFTB")
    args = parser.parse_args()
    if args.trainer == "simple":
        from trainers.simple import SimpleTrainer
        trainer = SimpleTrainer(args.conf)
    elif args.trainer == "PF":
        from old.pushforward import PFTrainer
        trainer = PFTrainer(args.conf)
    elif args.trainer == "PFTB":
        from trainers.PFTB import PFTBTrainer
        trainer = PFTBTrainer(args.conf)
    else:
        raise ValueError("Trainer not set")
    trainer.train()