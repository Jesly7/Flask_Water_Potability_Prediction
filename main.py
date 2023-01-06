import pandas as pd
from src.data_preprocessing import data_clean
from src.imbalance import data_balanced
from src.model import algo

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('water_potability.csv')
cleaned_data=data_clean(df)
print(cleaned_data)
balanced_data=data_balanced()
algo(RandomForestClassifier())
