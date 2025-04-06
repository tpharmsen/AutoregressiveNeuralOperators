import argparse
import yaml
import os

class DotDict(dict):
    def __init__(self, mapping=None):
        super().__init__()
        mapping = mapping or {} 
        for key, value in mapping.items():
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not in config")

    def __setattr__(self, key, value):
        self[key] = value

def load_yaml_as_dotdict(filepath):
    with open(filepath, "r") as file:
        data = yaml.safe_load(file) or {}  # for if yaml empty lined
    return DotDict(data)

def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CB", type=str, default="std.yaml")
    parser.add_argument("--CD", type=str, default="std.yaml")
    parser.add_argument("--CM", type=str, default="std.yaml")
    parser.add_argument("--CT", type=str, default="std.yaml")
    parser.add_argument("--trainer", type=str, default="STT")
    args = parser.parse_args()

    if os.path.exists("conf/base/" + args.CB):
        cb = load_yaml_as_dotdict("conf/base/" + args.CB)
    else:
        raise FileNotFoundError(f"Config file {args.CB} not found.")
    if os.path.exists("conf/data/" + args.CD):
        cd = load_yaml_as_dotdict("conf/data/" + args.CD)
    else:
        raise FileNotFoundError(f"Config file {args.CD} not found.")
    if os.path.exists("conf/model/" + args.CM):
        cm = load_yaml_as_dotdict("conf/model/" + args.CM)
    else:
        raise FileNotFoundError(f"Config file {args.CM} not found.")
    if os.path.exists("conf/training/" + args.CT):
        ct = load_yaml_as_dotdict("conf/training/" + args.CT)
    else:
        raise FileNotFoundError(f"Config file {args.CT} not found.")

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
        from trainers.STT import STT
        trainer = STT(cb, cd, cm, ct)
    else:
        raise ValueError("Trainer not set")

    trainer.train()