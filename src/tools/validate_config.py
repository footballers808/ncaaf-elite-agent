import sys, yaml, pathlib

REQUIRED = [
  ("form_games", int),
  ("pace_games", int),
  ("injury_window_days", int),
  ("model", dict),
]

def main():
    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
    for key, typ in REQUIRED:
        if key not in cfg:
            print(f"missing required config: {key}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(cfg[key], typ):
            print(f"bad type for {key}, expected {typ}", file=sys.stderr)
            sys.exit(2)
    print("config ok")

if __name__ == "__main__":
    main()
