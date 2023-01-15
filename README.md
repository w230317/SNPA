# SNPA
Stock Network Portfolio Allocation (SNPA)

```python
# import requirements
import pandas as pd  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import numpy as np  
import networkx as nx 
from datetime import datetime, timedelta  
from random import randint, random  
from sklearn.model_selection import RandomizedSearchCV  
from xgboost import XGBRegressor  
import os  

# if tested on google colab you need to install the yfinance package
try:
  import yfinance as yf
except:
  !pip install yfinance
  import yfinance as yf

import pandas_datareader.data as pdr
yf.pdr_override()

# refers to the SNPA.py file
from SNPA import * 

# stock example file
stock =  ((pd.read_csv('https://raw.githubusercontent.com/w230317/SNPA/main/data/stockB3.csv', header=None, usecols=[0])))
stock = stock.sample(60, random_state=42)

# call class
snpa = SPNA()

# Step 1 - Generate database from Yahoo Finance. Pay attention to the date format dd-mm-yyyy
# An array with asset names can also be used, example: snpa.gen_asset_data(['BBAS3.SA','ITUB4.SA','MGLU3.SA'], '01-01-2010', '01-01-2021')
snpa.gen_asset_data(stock, '01-01-2010', '01-01-2021')

# Step 2:
# Load  data
df_monthly_return = snpa.df_from_csv('snpa_df_monthly_return.csv')

# Define the period that the SNPA will use as well as the result date (if backtest)
# The 'forecast' hyperparameter specifies whether the algorithm will use XGBoost to forecast asset values
df, backtest = snpa.df_split_period(df_monthly_return, '2010-02-01', '2020-02-01', '2020-03-01', forecast=False)

# Run SNPA. Best parameters for forecast=True : lambda_p=0.6,lambda_n=-0.2
df_portfolio, df_edges = snpa.snpa(df,lambda_p=0.8,lambda_n=-0.5,k=4000, forecast=False)

# Get results (S, w)
print(df_portfolio)

# Get backtest
print(snpa.portfolio_backtest(backtest, df_portfolio))
```
