from classifier import rfClassifier
import os
import argparse
import pandas as pd
import pickle
import wandb
from evaluate import evaluate_model

def read_file(data_dir, split):
    """Lee un archivo pickle desde el directorio especificado y lo carga.
    Parameters
    ----------
    data_dir : str
        Directorio donde se encuentra el archivo.
    split : str
        Nombre del archivo (sin extensión) a leer.
    Returns
    -------
    object
        Objeto cargado desde el archivo pickle.
    """

    filename = split + ".pkl"
    with open(os.path.join(data_dir, filename), "rb") as f:
        result = pickle.load(f)
    return result




def train_and_log(config={},experiment_id='99',model_name="RandomForest", model_description="Simple RandomForest Classifier"):
    """Entrena un modelo y registra el modelo entrenado y las métricas en W&B.
    Parameters
    ----------
    config : dict
        Configuración del modelo.
    experiment_id : str, optional
        ID del experimento (default: '99').
    model_name : str, optional
        Nombre del modelo (default: "RandomForest").
    model_description : str, optional
        Descripción del modelo (default: "Simple RandomForest Classifier").
    """
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

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained RandomForest model",
            metadata=dict(model_config))

        name_artifact_model = f"trained_model_{model_name}.pkl"

        with model_artifact.new_file(f"./model/{name_artifact_model}", mode="wb") as file:
            pickle.dump(model, file)

        train_metrics = evaluate_model(model, training_dataset, training_labels, prefix="training_")
        val_metrics = evaluate_model(model, validation_dataset, validation_labels, prefix="validation_")

        run.summary.update(train_metrics)
        run.summary.update(val_metrics)

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)






if __name__ == "__main__":
    # training and evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"

    train_and_log(experiment_id=id,model_name="RandomForest", model_description="Simple RandomForest Classifier")    

