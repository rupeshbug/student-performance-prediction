import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

# DataTransformationConfig provides input to the DataTransformation component
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation()
        
    def get_data_transformer_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
            
            numerical_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ],
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read the train and test data")
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]
            
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]
        except:
            pass

