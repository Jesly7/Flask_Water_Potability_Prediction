import pandas as pd
import numpy as np
from pathlib import Path


def data_clean(df):
    df=df.fillna(df.median())
    for column in df.columns[:-1]:
        Q1=df[column].quantile(0.25)
        Q3=df[column].quantile(0.75)
        IQR=Q3-Q1
        lower_range=Q1-(IQR*1.5)
        upper_range=Q3+(IQR*1.5)
        df[column]=np.where(df[column]<lower_range,lower_range,df[column])
        df[column]=np.where(df[column]>upper_range,upper_range,df[column])
        x=df.drop('Potability',axis=1)
        y=df['Potability']
        p=Path('processed_data\processed_label.csv')
        x.to_csv(p,index=False)
        p=Path('processed_data\processed_class.csv')
        y.to_csv(p,index=False)
        
