import datetime
import os
import sys

import pandas as pd

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.plotting_ssusi_edr import SUSSI_Plotter
from utils.omni_data import OMNI_Data

load_dotenv()

START_DATE = "2011-01-01"
END_DATE = "2014-12-31"

AE_CUTOFF = 200

drive_path = os.getenv("DRIVE_PATH")
omni = OMNI_Data()
plotter = SUSSI_Plotter(drive_path, end_file="figures_non_labeled")

ae_df = omni.get_df_data(["AE1800"], START_DATE, END_DATE)
ae_df_filtered = ae_df[ae_df["AE1800"] > AE_CUTOFF]
ae_df_filtered["Date"] = pd.to_datetime(ae_df_filtered["Epoch"]).dt.date
high_indesity_dates = list(set(ae_df_filtered["Date"]))


ae_dict = plotter.build_ae_index(ae_df)
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

date_chunks = list(chunks(high_indesity_dates, 30))
for i, date_chunk in enumerate(date_chunks):
    print(f"Processing chunk {i + 1} of {len(date_chunks)}")
    plotter.plot_midnight_passes_from_dates(date_chunk, False, AE_CUTOFF)
    plotter.save_metadata_csv()
