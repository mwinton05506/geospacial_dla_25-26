from cdasws import CdasWs
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

class OMNI_Data:
    def __init__(self):
        self.cdas = CdasWs()
        self.dataset = "OMNI2_H0_MRG1HR"
        self.vars_available = self.cdas.get_variables(self.dataset)
    
    def get_df_data(self, vars_to_get, start, end):
        data = self.cdas.get_data(
            dataset = self.dataset,
            time0 = start,
            time1 = end,
            variables = vars_to_get
        )
        df = pd.DataFrame()
        if data[0]["http"]["status_code"] != 200:
            raise Exception(f"Data Response Error. Status Code: {data[0]["http"]["status_code"]}")
        for key, _ in data[1].items():
            df[key] = data[1][key]
        return df