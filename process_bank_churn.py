import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training inputs, validation inputs,
        training targets, and validation targets.
    """
    input_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    target_col = 'Exited'
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        df[input_cols], df[target_col], test_size=0.2, random_state=12, stratify=df[target_col])
    return train_inputs, val_inputs, train_targets, val_targets

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, List[str]]:
    """
    Scales numeric features using MinMaxScaler.

    Args:
        train_inputs (pd.DataFrame): Training data inputs.
        val_inputs (pd.DataFrame): Validation data inputs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, List[str]]: Scaled training and validation inputs,
        the scaler used, and the list of numeric columns.
    """
    asis_col = ['id']
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.difference(asis_col).tolist()
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler, numeric_cols

def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str], List[str]]:
    """
    Encodes categorical features using OneHotEncoder.

    Args:
        train_inputs (pd.DataFrame): Training data inputs.
        val_inputs (pd.DataFrame): Validation data inputs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str], List[str]]: Encoded training and validation inputs,
        the encoder used, the list of categorical columns, and the list of encoded column names.
    """
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoder,  encoded_cols, categorical_cols

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Preprocesses the input dataframe by dropping unnecessary columns, splitting the dataset,
    scaling numeric features, and encoding categorical features.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]: The preprocessed training inputs, validation inputs,
        training targets, validation targets, and the list of model columns.
    """
    df = df.drop(columns=['CustomerId', 'Surname'])

    train_inputs, val_inputs, train_targets, val_targets = split_dataset(df)
    train_inputs, val_inputs, scaler, numeric_cols = scale_numeric_features(train_inputs, val_inputs)
    train_inputs, val_inputs, encoder,  encoded_cols, categorical_cols = encode_categorical_features(train_inputs, val_inputs)

    model_cols = train_inputs.select_dtypes(include=[np.number, 'float64']).columns.tolist() + encoded_cols
    X_train = train_inputs[model_cols]
    X_val = val_inputs[model_cols]

    return X_train, X_val, train_targets, val_targets, model_cols, scaler, encoder, numeric_cols, encoded_cols, categorical_cols 

def get_processed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Wrapper function to preprocess data and get training and validation sets.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]: The processed training inputs, validation inputs,
        training targets, validation targets, and the list of model columns.
    """
    return preprocess_data(df)

def preprocess_new_data(new_df: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder, 
                        numeric_cols: List[str], categorical_cols: List[str], encoded_cols: List[str], 
                        model_cols: List[str]) -> pd.DataFrame:
    """
    Preprocess new data to prepare it for prediction using the same preprocessing steps
    applied to the training data.

    Args:
        new_df (pd.DataFrame): The new dataframe to preprocess.
        scaler (MinMaxScaler): The scaler used to scale numeric features in the training data.
        encoder (OneHotEncoder): The encoder used to encode categorical features in the training data.
        numeric_cols (List[str]): The list of numeric columns to be scaled.
        categorical_cols (List[str]): The list of categorical column to be encoded.
        encoded_cols (List[str]): The list of encoded column names after applying OneHotEncoder.
        model_cols (List[str]): The list of column names that the model expects.

    Returns:
        pd.DataFrame: The preprocessed new dataframe ready for prediction.
    """
    new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    new_df[encoded_cols] = encoder.transform(new_df[categorical_cols])
    X_test = new_df[numeric_cols + encoded_cols]
    
    # Упорядкування колонок згідно з model_cols
    X_test = X_test[model_cols]
    
    return X_test