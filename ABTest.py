
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_exploration():
    cookie_df = pd.read_excel('/data/cookie_cats.csv')
    return cookie_df

df = data_exploration()
print(df.head())