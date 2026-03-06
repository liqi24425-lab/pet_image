import json
import urllib.parse
import urllib.request

BASE = "https://datasets-server.huggingface.co"


def get_json(path: str, params: dict):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    # Read-only probe template for future HF experiment dataset.
    # Replace dataset name when a registry dataset is created on Hub.
    dataset = "openai/gsm8k"  # placeholder public dataset for connectivity check
    print("is-valid:")
    print(get_json("/is-valid", {"dataset": dataset}))
    print("splits:")
    splits = get_json("/splits", {"dataset": dataset})
    print({"dataset": dataset, "num_splits": len(splits.get("splits", []))})


if __name__ == "__main__":
    main()
