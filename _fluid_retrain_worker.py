"""_fluid_retrain_worker.py — FluidPipeline から spawn される再学習ワーカー。"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--triggered-by", default="fluid_pipeline")
args = parser.parse_args()

try:
    from retraining_pipeline import run_retraining
    result = run_retraining(
        triggered_by=args.triggered_by,
        db_path="data/lease_data.db",
        model_dir="models/",
    )
    # モデル更新完了通知
    if result.get("model_updated"):
        from fluid_pipeline import FluidPipeline
        FluidPipeline().on_model_updated(result)
    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    print(f"[_fluid_retrain_worker] error: {e}", file=sys.stderr)
