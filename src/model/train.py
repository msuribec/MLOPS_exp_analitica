from classifier import rfClassifier
import os
import argparse
import wandb

def read(data_dir, split):
    filename = split + ".csv"
    df = pd.read_csv(os.path.join(data_dir, filename))
    return df



def train_and_log(config,experiment_id='99'):

    seed = int(os.environ["SEED"])
    project_name = os.environ["PROJECT_NAME"]
    dataset_name = os.environ["DATASET_NAME"]

    with wandb.init(
        project=project_name, 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model", config=config) as run:
        config = wandb.config
        data = run.use_artifact(f'{dataset_name}-preprocess:latest')
        data_dir = data.download()

        training_dataset =  read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        model_artifact = run.use_artifact("RandomForest:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "model/initialized_model_RandomForest.pkl")
        model_config = model_artifact.metadata
        config.update(model_config)



        # model = rfClassifier(model_config)
        # model.load_state_dict(torch.load(model_path))
        # model = model.to(device)
 
        # train(model, train_loader, validation_loader, config)

        # model_artifact = wandb.Artifact(
        #     "trained-model", type="model",
        #     description="Trained NN model",
        #     metadata=dict(model_config))

        # torch.save(model.state_dict(), "trained_model.pth")
        # model_artifact.add_file("trained_model.pth")
        # wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"

    model = train_and_log(train_config, id)       
