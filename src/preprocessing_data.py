import polars as pl
import numpy as np
import joblib 
import os

current_dir = os.path.dirname(__file__)

target_encoder = joblib.load(os.path.join(current_dir,'pkl/target_encoder.pkl'))
xgboost = joblib.load(os.path.join(current_dir,'pkl/xgboost.pkl'))

columns = [
        "MEM_GENDER", "MEM_AGE", "PAYER_LOB", 
        "PRIMARY_CHRONIC_CONDITION_ROLLUP_ID", "DIAG_CCS_1_LABEL", 
        "MS_DRG", "DIAG_CCS_2_LABEL", "ADM_SRC", 
        "ADM_TYPE", "DIS_STAT", "State_HI", 
        "State_MI", "State_WA", "State_CA"
    ]  


class dataFrame:
    def __init__(self, data):
        self.data = pl.DataFrame({columns[i]:data[i] for i in range(len(columns))})
    def _encode_data(self):
        return target_encoder.transform(x = self.data)
    def predict(self):
        converted_data = self._encode_data().to_numpy()
        predicted = xgboost.predict(converted_data)
        return predicted