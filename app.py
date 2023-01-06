from flask import Flask, render_template, request, make_response
import numpy as np
import pandas as pd
import joblib as jb
from pathlib import Path

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ph=(request.form['ph'])
    Hardness=(request.form['Hardness'])
    Solids=(request.form['Solids'])
    Chloramines=(request.form['Chloramines'])
    Sulfate=(request.form['Sulfate'])
    Conductivity=(request.form['Conductivity'])
    Organic_carbon=(request.form['Organic_carbon'])
    Trihalomethanes=(request.form['Trihalomethanes'])
    Turbidity=(request.form['Turbidity'])

    arr=np.array([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]])

    model=jb.load(Path('model\model_rfc.pkl'))
    result=model.predict(arr)
    if result ==1:
        return render_template('after.html',data='Its safe for drink')
    else:
        return render_template('after.html',data='Not a Drinking Water')


@app.route('/uploader',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file=request.files['file']
        df=pd.DataFrame(pd.read_csv(file))
        model=jb.load(Path('model\model_rfc.pkl'))
        result=pd.DataFrame(model.predict(df))
        result.columns=['Result']
        result=result.replace({1:'Drinking water' , 0:'Not a Drinking water'})
        response=make_response(result.to_csv)
        response.headers['Content-Disposition']='attachment; filename=Water Potability Prediction.csv'
        response.mimetype='text/csv'
        return response




if __name__ == '__main__':
    app.run(debug=True)