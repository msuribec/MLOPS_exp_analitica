from classifier import rfClassifier

import os
import argparse
import wandb


def build_model_and_log(config, model, model_name="RandomForest", model_description="Simple RandomForest Classifier"):

    project_name = os.environ["PROJECT_NAME"]
    dataset_name = os.environ["DATASET_NAME"]

    with wandb.init(project=project_name, 
        name=f"initialize Model ExecId-{args.IdExecution}", 
        job_type="initialize-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_model_{model_name}.pkl"

        with model_artifact.new_file(f"./model/{name_artifact_model}", mode="wb") as file:
            pickle.dump((df_x, labels), file)

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"

    # Check if the directory "./model" exists
    if not os.path.exists("./model"):
        # If it doesn't exist, create it
        os.makedirs("./model")

    seed = int(os.environ["SEED"])
    model_config = {
        "criterion" : "gini",
        "n_estimators" : 1750,
        "max_depth": 7,
        "min_samples_split": 6,
        "min_samples_leaf": 6,
        "random_state": seed,
        "n_jobs": -1
    }

    model = rfClassifier(**model_config)

    build_model_and_log(model_config, model, "linear","Simple Linear Classifier")