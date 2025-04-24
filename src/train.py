import argparse
import yaml
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
#os.environ["WANDB_SILENT"] = "true"

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
    parser.add_argument("--CB", type=str, default="std")
    parser.add_argument("--CD", type=str, default="std")
    parser.add_argument("--CM", type=str, default="std")
    parser.add_argument("--CT", type=str, default="std")
    parser.add_argument("--trainer", type=str, default="MTT")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if os.path.exists("conf/base/" + args.CB + ".yaml"):
        cb = load_yaml_as_dotdict("conf/base/" + args.CB + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CB}.yaml not found.")
    if os.path.exists("conf/data/" + args.CD + ".yaml"):
        cd = load_yaml_as_dotdict("conf/data/" + args.CD + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CD}.yaml not found.")
    if os.path.exists("conf/model/" + args.CM + ".yaml"):
        cm = load_yaml_as_dotdict("conf/model/" + args.CM + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CM}.yaml not found.")
    if os.path.exists("conf/training/" + args.CT + ".yaml"):
        ct = load_yaml_as_dotdict("conf/training/" + args.CT + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CT}.yaml not found.")


    if args.out != None:
        cb.folder_out = args.out.replace("/", "") + "/"
        #print('args flag')
    os.makedirs(cb.save_path + cb.folder_out, exist_ok=True)
    if args.out != None:
        cb.wandb_name = args.out
        

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
    elif args.trainer == "MTT":
        from trainers.MTT import MTT
        trainer = MTT(cb, cd, cm, ct)
    else:
        raise ValueError("Trainer not set")


    trainer.train()