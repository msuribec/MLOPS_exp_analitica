from classifier import rfClassifier
import os
import argparse
import pandas as pd
import pickle
import wandb

def read_file(data_dir, split):
    filename = split + ".pkl"
    with open(os.path.join(data_dir, filename), "rb") as f:
        result = pickle.load(f)
    return result




def train_and_log(config={},experiment_id='99',model_name="RandomForest", model_description="Simple RandomForest Classifier"):

    seed = int(os.environ["SEED"])
    project_name = os.environ["PROJECT_NAME"]
    dataset_name = os.environ["DATASET_NAME"]

    with wandb.init(
        project=project_name, 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model", config=config) as run:
        config = wandb.config

        preprocess_data_artifact = run.use_artifact(f'{dataset_name}-preprocess:latest')

        # 📥 if need be, download the artifact
        preprocess_dataset = preprocess_data_artifact.download(root="./data/artifacts/")

        training_data =  read_file(preprocess_dataset, "training")
        training_dataset, training_labels = training_data
        validation_data = read_file(preprocess_dataset, "validation")
        validation_dataset, validation_labels = validation_data

        model_artifact = run.use_artifact("RandomForest:latest")
        model_dir = model_artifact.download()
        model_config = model_artifact.metadata
        config.update(model_config)

        model = read_file(model_dir, "model/initialized_model_RandomForest")

        model.fit(training_dataset, training_labels)

        trained_model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained RandomForest model",
            metadata=dict(model_config))

        name_artifact_model = f"trained_model_{model_name}.pkl"

        with model_artifact.new_file(f"./model/{name_artifact_model}", mode="wb") as file:
            pickle.dump(model, file)

        wandb.save(name_artifact_model)

        run.log_artifact(trained_model_artifact)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"

    model = train_and_log(experiment_id=id,model_name="RandomForest", model_description="Simple RandomForest Classifier")       
