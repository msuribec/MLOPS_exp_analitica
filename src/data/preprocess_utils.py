import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin # for creating custom transformers

class BinaryMapTransformer(BaseEstimator, TransformerMixin):
    """Transformer que aplica mapeos personalizados para convertir variables categóricas binarias en valores numéricos."""
    def __init__(self, mapping:dict):
        """Inicializa el transformer con el diccionario de mapeos."""
        self.mapping = mapping

    def fit(self, X, y=None):
        """No realiza ajuste, solo retorna self."""
        return self

    def transform(self, X):
        """Aplica el mapeo definido a cada columna binaria del DataFrame."""
        X = pd.DataFrame(X).copy()
        self.feature_names_out_ = list(self.mapping.keys())
        i = 0
        for col, mapping in self.mapping.items():
            X.iloc[:, i] = X.iloc[:, i].map(mapping)
            i = i+1
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    """Transformer que elimina las columnas especificadas del DataFrame."""
    def __init__(self, cols):
        """Inicializa el transformer con la lista de columnas a eliminar."""
        self.cols = cols
    def fit(self, X, y=None):
        """No realiza ajuste, solo retorna self."""
        return self
    def transform(self, X):
        """Elimina las columnas especificadas del DataFrame."""
        X = pd.DataFrame(X).copy()
        self.feature_names_out_ = [c for c in X.columns if c not in self.cols]
        return X[self.feature_names_out_]
    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas después de la transformación."""
        return np.array(self.feature_names_out_, dtype=object)


class CastToInt(BaseEstimator, TransformerMixin):
    """Transformer que convierte las columnas numéricas especificadas a tipo int64, manteniendo los valores NaN."""
    def __init__(self, columns):
        """Inicializa el transformer con la lista de columnas a convertir a int."""
        self.columns = columns

    def fit(self, X, y=None):
        """Guarda los nombres de las columnas y retorna self."""
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        """Convierte las columnas especificadas a tipo int64."""
        X = X.copy()
        X[self.columns] = X[self.columns].astype(np.int64)
        return X

        return X

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas después de la transformación."""
        return np.array(self.feature_names_out_, dtype=object)

class CastToFloat(BaseEstimator, TransformerMixin):
    """Transformer que convierte las columnas numéricas especificadas a tipo float64, manteniendo los valores NaN."""
    def __init__(self, columns):
        """Inicializa el transformer con la lista de columnas a convertir a float."""
        self.columns = columns

    def fit(self, X, y=None):
        """Guarda los nombres de las columnas y retorna self."""
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        """Convierte las columnas especificadas a tipo float64."""
        X = X.copy()
        X[self.columns] = X[self.columns].astype('float64')
        return X

        return X

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas después de la transformación."""
        return np.array(self.feature_names_out_, dtype=object)


class OrganizeColumns(BaseEstimator, TransformerMixin):
    """Transformer que renombra las columnas de un DataFrame a snake_case."""
    def fit(self, X, y=None):
        """Ajusta el transformer. No realiza ninguna operación de ajuste."""
        return self

    def transform(self, X):
        """Renombra las columnas del DataFrame a snake_case y elimina dobles guiones bajos."""
        X = X.copy()
        self.feature_names_out_ = sorted(X.columns)
        X.columns = self.feature_names_out_
        return X
    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas después de la transformación."""
        return np.array(self.feature_names_out_, dtype=object)

def add_features(df,check_cols):
  """Función para agregar nuevas características al DataFrame.
    Actualmente, esta función verifica la presencia de ciertas columnas y las agrega si no existen, inicial
    lizándolas con un valor predeterminado de 0.0.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame al que se agregarán las nuevas características.
    check_cols : list
        Lista de nombres de columnas que se verificarán y agregarán si no existen en el DataFrame.
    Returns
    -------
    pd.DataFrame 
        DataFrame con las nuevas características agregadas.
    """
  for col in check_cols:
    if col not in df.columns:
      df[col] = 0.0
  return df