import pandas as pd
from pathlib import Path
import joblib as jb
import json

def algo(model):
    x_test=pd.read_csv(Path(r'Balanced_processed_data\x_test.csv'))
    x_train=pd.read_csv(Path(r'Balanced_processed_data\x_train.csv'))
    y_test=pd.read_csv(Path(r'Balanced_processed_data\y_test.csv'))
    y_train=pd.read_csv(Path(r'Balanced_processed_data\y_train.csv'))
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print('..................................................\n',model,'Report\n.................................................')
    train_score=model.score(x_train,y_train)
    test_score=model.score(x_test,y_test)
    jb.dump(model,Path('model\model_rfc.pkl'))
    path=Path('report\metrics.json')
    with open(path,'w') as f:

        scores = {
            "test_score" :test_score,
            "train_score" :train_score,
            }
        json.dump(scores,f)