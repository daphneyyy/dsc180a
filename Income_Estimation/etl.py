import pandas as pd
import pyarrow 
def get_data():
    parquet_file1 = 'Transacation_inflows_with_date_3k.pqt'

    inflow = pd.read_parquet(parquet_file1, engine='pyarrow')
    return inflow
