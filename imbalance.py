from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

def data_balanced():
    x=pd.read_csv('processed_data\processed_label.csv')
    y=pd.read_csv('processed_data\processed_class.csv')
    os=RandomOverSampler()
    X,Y=os.fit_resample(x,y)
    counter=Counter(Y)
    print(counter)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=345)
    x_train.to_csv(Path(r'Balanced_processed_data\x_train.csv'),index=False)
    x_test.to_csv(Path(r'Balanced_processed_data\x_test.csv'),index=False)
    y_train.to_csv(Path(r'Balanced_processed_data\y_train.csv'),index=False)
    y_test.to_csv(Path(r'Balanced_processed_data\y_test.csv'),index=False)