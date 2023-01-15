# requirements
# import pandas as pd
# import numpy as np
# import networkx as nx
# from datetime import datetime, timedelta
# from random import randint, random
# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBRegressor

# import os
# try:
#   import yfinance as yf
# except:
#   !pip install yfinance
#   import yfinance as yf

# import pandas_datareader.data as pdr
# yf.pdr_override()

class SPNA:
  
   def __init__(self):
    pass
  
  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
  
  def portfolio_backtest(self, test, portfolio):
    try:
      date = test.index.astype(str)[0]
      t = test.T.reset_index()
      t['date'] = date
      t.columns = ['S', 'return', 'dt']
      t = pd.merge(portfolio,
                      t,
                      on='S', 
                      how='left').dropna()

      t['proportional_return'] = t['w'] * t['return']                 
      t = t.groupby('dt').agg(wealth=('proportional_return', 'sum')).reset_index()
      return t

    except:
      print('ERROR: error while trying portfolio forecating....')

  def df_from_csv(self, file_csv):

    if self.gen_asset_data_error():
      return None


    df = pd.read_csv(file_csv)
    retornos_mensal = df
    df.index = pd.to_datetime(df['Date'])
    df = df.drop('Date', axis=1) 
    return df  


  def df_split_period(self, df, start_date, end_date, backtest_date, forecast=False):
    
    if self.gen_asset_data_error():
      return None, None

    train = pd.DataFrame(df[df.index < backtest_date])
    train = pd.DataFrame(train[train.index >= start_date])
    train = pd.DataFrame(train[train.index <= end_date])

    test = pd.DataFrame(df[df.index == backtest_date])

    if len(test)==0:
      print('WARN: the reported forecast date does not exist in the dataset...')
      test = pd.DataFrame() # return empty dataframe
    
    
    df_final = pd.DataFrame()

    if forecast:
      lags = 6  
      df_daily_volume = snpa.df_from_csv('snpa_df_daily_volume.csv')
      dfvol = pd.DataFrame(df_daily_volume[df_daily_volume.index < backtest_date])
      dfvol = pd.DataFrame(dfvol[dfvol.index >= start_date])
      dfvol = pd.DataFrame(dfvol[dfvol.index <= end_date])

      df_daily_close = snpa.df_from_csv('snpa_df_daily_close.csv')
      dfpric = pd.DataFrame(df_daily_close[df_daily_close.index < backtest_date])
      dfpric = pd.DataFrame(dfpric[dfpric.index >= start_date])
      dfpric = pd.DataFrame(dfpric[dfpric.index <= end_date])

      
      for c in df.columns:
        print('Forecasting ' + c)
        
        dfc = pd.DataFrame(dfpric[c].values).rename(columns={0: 'price'})
        dfv = pd.DataFrame(dfvol[c].values).rename(columns={0: 'vol'})
        dfc.dropna(inplace=True)
        dfc['r_log'] = np.log(dfc / dfc.shift(1)).fillna(0)
        dfc['vol_log'] = np.log(dfv / dfv.shift(1)).fillna(0)
        dfc.replace([np.inf, -np.inf], 0, inplace=True)
        dfc.index=dfpric.index

        indM = pd.date_range(start=dfc.index[0],end=dfc.index[-1], freq='MS')
        df1 = pd.DataFrame(index = indM.strftime('%Y-%m'))
        l1 = []
        l2 = []  # volume
        for f in range(len(indM)):
            strAux = indM[f].strftime('%Y-%m')
            l1.append(dfc.loc[strAux]['r_log'].sum())
            l2.append(dfc.loc[strAux]['vol_log'].std())   
        cols = []
        df1['mr'] = l1
        df1['mvol'] = l2    
        df1['mvol'] = df1['mvol'].fillna(0.0) 
        df1['target'] = df1['mr'].shift(-1)  # next month return
        features = ['mr','mvol'] 
        for f in features:
            for lag in range(0, lags + 1):
                col = f'{f}_lag_{lag}'
                df1[col] = df1[f].shift(lag)
                cols.append(col)
        df1 = df1.tail(-lags).fillna(0)
        df_result = pd.DataFrame()

        trainSize = lags

        params = {'n_estimators': [10, 50, 100, 500, 1000, 2000],
                  'max_depth': [2,3,4,5,6,7,8,9,10,15],
                  'eta': [0.0001, 0.001, 0.01, 0.1, 1.0]}

        dfx = df1
        for i in range(0,len(dfx)-(trainSize)):
                x_train = np.array(dfx.iloc[i:trainSize+i,3:])
                y_train = np.array(dfx.iloc[i:trainSize+i,2])
                x_test = np.array(dfx.iloc[[trainSize+i],3:]) 
                y_test = np.array([dfx.iloc[trainSize+i,2]])

                model = XGBRegressor(silent=True, verbosity=1)
                selector = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, scoring= 'neg_root_mean_squared_error', verbose=0)
                selector.fit(x_train, y_train)
                
                yhat_test = (selector.predict(x_test))
                result = ({
                  'month'  :  dfx.iloc[[trainSize+i],3:].index[0],
                  'y_test' :  y_test,
                  'yhat_test' : yhat_test
                })

                df_result = pd.concat([df_result, pd.DataFrame(result)])

        df_final = pd.concat([df_final, df_result['yhat_test'].tail(1)], axis=1)
      
      df_final.index=[pd.to_datetime(train.tail(1).index[0] + pd.DateOffset(months=1)).strftime('%Y-%m-%d')]

      df_final.columns = train.columns

      train = pd.concat([train, df_final], axis=0)

      train.index = [pd.to_datetime(i).strftime('%Y-%m-%d') for i in train.index]
        

    return train, test


  def snpa(self, df,  k=1000, lambda_p=0.5, lambda_n=-0.5, q=0, forecast=False):
    
    if self.gen_asset_data_error():
      return None, None

    if forecast:
      alpha = 0.1
      RA = ((df + 1) * alpha).cumprod()-1  
      RA = RA.tail(1)
      cw = RA.T
      cw.columns=['cw']      
    else:
    #calculate cumulated wealth
      RA = (df + 1).cumprod()-1  
      RA = RA.tail(1)
      cw = RA.T
      cw.columns=['cw']

    # daily price of assets
    df_corr = pd.read_csv('snpa_df_daily_close.csv')
    df_corr.index = pd.to_datetime(df_corr['Date'])
    df_corr = df_corr.drop('Date', axis=1)
    df_corr = df_corr.loc[(df_corr.sum(axis=1) != 0), (df_corr.sum(axis=0) != 0)]

    # correlation matrix M
    M = df_corr.fillna(0).corr()
    np.fill_diagonal(M.values, 0)
    # exclude rows/columns with zero deviation
    excluir = pd.DataFrame(M.std()==0).reset_index().rename(columns={0: "excluir"})
    excluir = excluir[excluir.excluir==True]
    M.drop(np.array(excluir['index']), inplace=True, axis=0)
    M.drop(np.array(excluir['index']), inplace=True, axis=1)
    
    # Build a network, G, through the matrix M, respecting the constraints λn and λp
    G = nx.Graph(M)
    edge_weights = nx.get_edge_attributes(G,'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w < lambda_p and w > lambda_n))
    G.remove_nodes_from(list(nx.isolates(G)))

    # exclusions (diag correlation = 0)
    ndf = pd.DataFrame(G.edges(data='weight'))
    ndf.columns = ['from','to','weight']
    ndf = ndf[ndf.weight != 0.0]
    xndf = ndf
    ndfi = xndf[['to','from','weight']]
    ndfi.columns = ['from','to','weight']
    xdf = xndf.append (ndfi)

    # add cw to data
    df = pd.merge(left=xdf, right=cw, how='inner', left_on='to', right_on=cw.index)

    # include degree of nodes
    degree = pd.DataFrame(df.groupby(['from']).size())
    degree.columns=['degree_from']
    df = pd.merge(left=df, right=degree, how='left', left_on='to', right_on=degree.index)

    df_visited = []
    edge_visited = pd.DataFrame()

    node_unique = pd.unique(df['from'])
    # randomly select the start node
    start_node = node_unique[ randint(0, len(node_unique)-1) ]
    a = start_node


    for i in range(0,  k):
      b = df[df['from']==a]

      b['prob'] = self.softmax(b['cw']).fillna(0.0)
      b = b.sort_values(by=['prob'], ascending=False)
      b['prob_ant'] = pd.DataFrame(b['prob'].cumsum().shift(fill_value=0))
      b['prob_prox'] =  b['prob'].cumsum()        

      z = random()

      b['sel'] =  (z >= b['prob_ant']) & (z < b['prob_prox'])

      # previous node
      a_ant = a  
      df_visited = np.append(df_visited, a_ant)

      a = np.array(b[b['sel']==True]['to'])[0]

      vdata = {"from" : [a_ant],
               "to" : [a],
               "visited" : 1,
              }
      edge_visited =  pd.concat([edge_visited, pd.DataFrame(vdata)], axis=0) 


    vst = pd.DataFrame(df_visited, columns=['S'])
    vst['ttl'] = 1
    
    
    if q > 0:

      visited = vst.groupby(by=["S"]).sum()
      visited['w'] = visited['ttl'] / vst.count()[0]
      

      x = visited.reset_index().sort_values(by='w', ascending=False)
      x = x['S'].unique()[:q]
      vst = vst[vst['S'].isin(x)]
      visited = vst.groupby(by=["S"]).sum()
      visited['w'] = visited['ttl'] / vst.count()[0]
            
    else:

      visited = vst.groupby(by=["S"]).sum()
      visited['w'] = round(visited['ttl'] / vst.count()[0], 2)
    
    visited = visited[visited.w > 0.0]

    r = visited.sort_values(by=['w'], ascending=False)

    r = r['w'].reset_index()

    return r, edge_visited


  def gen_asset_data_error(self):
    file_daily_close = 'snpa_df_daily_close.csv'
    if not os.path.exists(file_daily_close):    
      print('ERROR: you need to generate the asset data to continue...')
      return True
    else:
      return False

  def gen_asset_data(self, symbol, start_date, end_date):
      # generate asset data
      file_monthly_return = 'snpa_df_monthly_return.csv'
      file_daily_close = 'snpa_df_daily_close.csv'
      file_daily_volume = 'snpa_df_daily_volume.csv'

      # remove older files if exists

      if os.path.exists(file_monthly_return):
          os.remove(file_monthly_return)
      if os.path.exists(file_daily_close):
          os.remove(file_daily_close)
      if os.path.exists(file_daily_volume):
          os.remove(file_daily_volume)                    

      # convert list/dataframe of assets to array
      if type(symbol)==pd.core.frame.DataFrame:
        symbols = np.array(symbol[0])
      else:
        symbols = np.array(symbol)

      
      # df_monthly_return = pd.DataFrame()

      cols=[]
      
      # get data from yahoo fynance
      start = datetime.strptime(start_date, '%d-%m-%Y')
      end = datetime.strptime(end_date, '%d-%m-%Y')      

      df_daily_close  = pd.date_range(start, end , freq='D') 
      df_daily_close = [d.strftime('%Y-%m-%d') for d in df_daily_close]
      df_daily_close = pd.DataFrame(df_daily_close)
      df_daily_close.columns=['Date']
      df_daily_close.index = df_daily_close['Date']

      df_daily_volume  = pd.date_range(start, end , freq='D') 
      df_daily_volume = [d.strftime('%Y-%m-%d') for d in df_daily_volume]
      df_daily_volume = pd.DataFrame(df_daily_volume)
      df_daily_volume.columns=['Date']
      df_daily_volume.index = df_daily_volume['Date']      


      df_monthly_return = pd.date_range(start, end , freq='1M') 
      df_monthly_return = [d.strftime('%Y-%m-01') for d in df_monthly_return]
      df_monthly_return = pd.DataFrame(df_monthly_return)
      df_monthly_return.columns=['Date']
      df_monthly_return.index = df_monthly_return['Date']


 
      print(start, end)
      for s in symbols:

        try:
          print('Stock '+ s)
          df = pdr.get_data_yahoo(s, start=start, end=end)
          p = df['Adj Close'] # price
          p = p.fillna(0.0)
          p = p[p != 0]
          
          v = df['Volume'] # volume
          v = v.fillna(0.0)

          if (len(p)) > 0:

            monthly_return = pd.DataFrame(p.resample('M').ffill().pct_change().fillna(0.0))
            monthly_return.index = monthly_return.index.strftime("%Y-%m-01")
            df_monthly_return[s] = pd.DataFrame(monthly_return)
            df_monthly_return = df_monthly_return.fillna(0.0)

            p.index = [d.strftime('%Y-%m-%d') for d in p.index]          
            v.index = [d.strftime('%Y-%m-%d') for d in v.index]

            df_daily_close[s] = pd.DataFrame(p)
            df_daily_close = df_daily_close.fillna(0.0)

            df_daily_volume[s] = pd.DataFrame(v)
            df_daily_volume = df_daily_volume.fillna(0.0)


            cols.append(s)
        except:
          pass
          print('ERROR ' + s)
            
      df_monthly_return = df_monthly_return.drop('Date', axis=1) 
      df_daily_close = df_daily_close.drop('Date', axis=1) 
      df_daily_volume = df_daily_volume.drop('Date', axis=1) 

      df_daily_close = df_daily_close.loc[(df_daily_close.sum(axis=1) != 0), (df_daily_close.sum(axis=0) != 0)]
      df_daily_volume = df_daily_volume.loc[(df_daily_volume.sum(axis=1) != 0), (df_daily_volume.sum(axis=0) != 0)]

      df_monthly_return.to_csv(file_monthly_return)
      df_daily_close.to_csv(file_daily_close)
      df_daily_volume.to_csv(file_daily_volume)


      return 'Finished ' + str(len(cols)) +' of '+ str(len(symbols))
