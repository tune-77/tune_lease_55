import json
import sys
import traceback

from api.scoring_full import _run_full_scoring_api_locked


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m api.scoring_worker INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path, output_path = sys.argv[1], sys.argv[2]
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            inputs = json.load(f)
        result = _run_full_scoring_api_locked(inputs)
        payload = {"ok": True, "result": result}
    except BaseException as e:
        payload = {
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
