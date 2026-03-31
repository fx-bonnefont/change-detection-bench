import argparse
import mlflow
from src.data.feature_dataset import FeatureDataset
from src.models.baseline import BaselineCLS, BaselinePatches
from src.training.trainer import evaluate_baseline, log_results
from src.config import PRETRAINED_MODEL_NAMES

MODELS = {
    "baseline-cls": BaselineCLS,
    "baseline-all-patches": BaselinePatches,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline-cls", choices=MODELS.keys())
    parser.add_argument("--backbone_size", type=str, default="small", choices=PRETRAINED_MODEL_NAMES.keys())
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    model = MODELS[args.model]()
    dataset = FeatureDataset(args.backbone_size, "valid")

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(f"CD-BENCH_{args.model}")

    if model.trainable:
        pass
    else:
        results = evaluate_baseline(model, dataset, args.batch_size)
        with mlflow.start_run(run_name=args.backbone_size):
            mlflow.log_params({
                "model": args.model,
                "backbone_size": args.backbone_size,
            })
            log_results(results)

    print(f"Done. Results logged to MLFlow: {args.model}")


if __name__ == "__main__":
    main()
