from sklearn.model_selection import train_test_split
from pathlib import Path
import seaborn as sns
import pandas as pd
import argparse
import wandb
import os

def load(train_size: float = 0.8, seed=42):
    """
    Load the Titanic dataset (from seaborn) and split into train/val/test.

    Parameters
    ----------
    train_size : float, optional
        Proportion of the dataset used for the training split (default: 0.8).
        The remaining 20% is split equally into validation and test.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Tuple of (train_df, val_df, test_df).
    """
    # Load Titanic
    df = sns.load_dataset("titanic")

    # Binary target for classification
    target_col = "survived"
    df = df.dropna(subset=[target_col]).copy()

    # Simple split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=1.0 - train_size, random_state=seed, stratify=df[target_col]
    )
    # Split the remainder equally into val/test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df[target_col]
    )

    datasets = [train_df, val_df, test_df]

    return datasets

def load_and_log(id_execution: str | None = None):
    # 🚀 start a run, with a type to label it and a project it can call home

    if id_execution:
            print(f"IdExecution: {id_execution}")

    with wandb.init(
        project="MLOps-ExpAnalitica",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        seed = int(os.environ["SEED"])
        train_size = int(os.environ["TRAIN_SIZE"])

        datasets = load(train_size=train_size, seed=seed)

        train_df, val_df, test_df = datasets

        # Save locally
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_artifact = wandb.Artifact(
            "titanic-raw", type="dataset",
            description="raw TITANIC dataset, split into train/val/test",
            metadata={"source": "seaborn.load_dataset('titanic')",
                      "sizes": [len(dataset) for dataset in datasets]})

        raw_artifact.add_file(str(train_path))
        raw_artifact.add_file(str(val_path))
        raw_artifact.add_file(str(test_path))
        run.log_artifact(raw_artifact)

        run.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    load_and_log(id_execution=args.IdExecution)