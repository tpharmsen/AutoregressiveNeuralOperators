import argparse
import yaml


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CB", type=str, default="conf/base/std.yaml")
    parser.add_argument("--CD", type=str, default="conf/data/std.yaml")
    parser.add_argument("--CM", type=str, default="conf/model/std.yaml")
    parser.add_argument("--CT", type=str, default="conf/training/std.yaml")
    parser.add_argument("--CV", type=str, default="conf/validation/std.yaml")
    parser.add_argument("--trainer", type=str, default="PFTB")
    args = parser.parse_args()

    if os.path.exists(args.CB):
        cb = load_yaml(args.CB)
    else:
        raise FileNotFoundError(f"Config file {args.CB} not found.")
    if os.path.exists(args.CD):
        cd = load_yaml(args.CD)
    else:
        raise FileNotFoundError(f"Config file {args.CD} not found.")
    if os.path.exists(args.CM):
        cm = load_yaml(args.CM)
    else:
        raise FileNotFoundError(f"Config file {args.CM} not found.")
    if os.path.exists(args.CT):
        ct = load_yaml(args.CT)
    else:
        raise FileNotFoundError(f"Config file {args.CT} not found.")
    if os.path.exists(args.CV):
        cv = load_yaml(args.CV)
    else:
        raise FileNotFoundError(f"Config file {args.CV} not found.")

    if args.trainer == "simple":
        from trainers.simple import SimpleTrainer
        trainer = SimpleTrainer(args.conf)
    elif args.trainer == "PF":
        from old.pushforward import PFTrainer
        trainer = PFTrainer(args.conf)
    elif args.trainer == "PFTB":
        from trainers.PFTB import PFTBTrainer
        trainer = PFTBTrainer(args.conf)
    elif args.trainer == "STT":
        from trainers.simpleTransformerTrainer import simpleTransformerTrainer
        trainer = simpleTransformerTrainer(cb, cd, cm, ct, cv)
    else:
        raise ValueError("Trainer not set")

    trainer.train()