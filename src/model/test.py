
import os
import argparse
import pandas as pd
import pickle
import wandb
from evaluate import evaluate_model

def read_file(data_dir, split):
    filename = split + ".pkl"
    with open(os.path.join(data_dir, filename), "rb") as f:
        result = pickle.load(f)
    return result


def evaluate_and_log(config={},experiment_id='99',model_name="RandomForest", model_description="Simple RandomForest Classifier"):

    seed = int(os.environ["SEED"])
    project_name = os.environ["PROJECT_NAME"]
    dataset_name = os.environ["DATASET_NAME"]

    with wandb.init(
        project=project_name, 
        name=f"Eval Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="eval-model", config=config) as run:
        config = wandb.config
        preprocess_data_artifact = run.use_artifact(f'{dataset_name}-preprocess:latest')

        # 📥 if need be, download the artifact
        preprocess_dataset = preprocess_data_artifact.download(root="./data/artifacts/")

        test_data =  read_file(preprocess_dataset, "test")
        test_dataset, test_labels = test_data

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_config = model_artifact.metadata
        config.update(model_config)

        model = read_file(model_dir, "model/trained_model_RandomForest")

        test_metrics = evaluate_model(model, test_dataset, test_labels, prefix="test_")

        run.summary.update(test_metrics)




if __name__ == "__main__":
    # training and evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"


    evaluate_and_log(experiment_id=id,model_name="RandomForest", model_description="Simple RandomForest Classifier")   
