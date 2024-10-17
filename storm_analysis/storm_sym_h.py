import numpy as np
import pandas as pd
from datetime import datetime, time, date, timedelta

# Load the sym-h Excel file
df = pd.read_excel("data/sym_h/sym_h_may2024.xlsx")
spreadsheet_columns_as_arrays = df.values.T
sym_h = spreadsheet_columns_as_arrays[6]


pass