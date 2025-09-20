from typing import Dict
import os
import argparse
import wandb
import pandas as pd
import pickle
from preprocess_utils import (
    DropColumns,
    CastToInt,
    CastToFloat,
    BinaryMapTransformer,
    OrganizeColumns,
    add_features
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config # to set configuration options
from sklearn.compose import ColumnTransformer # for column transformations
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


def build_preprocess_pipeline(pipeline_config: Dict):

    drop_cols = pipeline_config.get("drop_cols", [])
    binary_cols = pipeline_config.get("binary_cols", [])
    bin_map = pipeline_config.get("bin_map", {})
    numeric_cols = pipeline_config.get("numeric_cols", [])
    onehot_cols = pipeline_config.get("onehot_cols", [])

    process_cols = binary_cols + numeric_cols  + onehot_cols

    coltx = ColumnTransformer(
            transformers=[
                # 1) Variables binarias: imputar con moda y mapear a 0/1
                ('binary', Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('binmap',  BinaryMapTransformer(mapping=bin_map))
                    # Después del mapeo no quedan NaNs, se mantienen como numéricas
                ]), binary_cols),

                # 2) Variables numéricas: imputar con mediana y escalar
                ('numeric', Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler()),
                ]),numeric_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
    one_hot_coltx = ColumnTransformer(
            transformers=[
                # 3) Variables categoricas
                ('onehot', Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), onehot_cols),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
    # Configuración para que el output del transformador sea un DataFrame de pandas
    set_config(transform_output="pandas")
    # Pipeline de preprocesamiento que aplica todas las transformaciones definidas
    preprocessor = Pipeline([
        ('drop_id', DropColumns(cols=drop_cols)), # Elimina la columna 'id'
        ("cast_float", CastToFloat(columns=numeric_cols)), # Convierte columnas numéricas a float
        ('by_type', coltx), # Aplica transformaciones por tipo de variable
        ('one_hot',one_hot_coltx),
        ("features", FunctionTransformer(add_features, validate=False)),
        ("cast_int", CastToInt(columns=binary_cols)), # Convierte columnas binarias a int,
        ("organize", OrganizeColumns()),

    ])

    return preprocessor


def preprocess(df:pd.DataFrame,config:Dict = {},target: str = 'label') -> pd.DataFrame:
    preprocessor = build_preprocess_pipeline(config)
    labels = df[target].values
    df = preprocessor.fit_transform(df)
    return df, labels

def preprocess_and_log(steps):

    with wandb.init(project="MLOps-ExpAnalitica",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed Titanic dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('titanic-raw:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["train", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pkl", mode="wb") as file:
                df_x, labels = processed_dataset.tensors
                pickle.dump((df_x, labels), file)


        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".csv"
    df = pd.read_csv(os.path.join(data_dir, filename))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    else:
        args.IdExecution = "testing console"


    TARGET = "survived"


    pre_processor_confg = {
        "drop_cols": ["class", TARGET, "alive",'embark_town'],
        "binary_cols": ["sex","adult_male", "alone"],
        "bin_map": {
            'sex': {'male': 1, 'female': 0},
            'adult_male': {True: 1, False: 0},
            'alone': {True: 1, False: 0},
        },
        "numeric_cols": ["pclass", "age", "sibsp", "parch", "fare"],
        "onehot_cols": ["embarked", "who", "deck"],
        "check_cols" : [ 'deck_A','deck_B','deck_C','deck_D','deck_E','deck_F','deck_G','embarked_C','embarked_Q','embarked_S','who_child','who_man','who_woman']
    }

    steps = {
        "config": pre_processor_confg,
        "target": TARGET
    }

    preprocess_and_log(steps)