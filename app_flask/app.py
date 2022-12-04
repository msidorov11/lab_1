from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict/<int:Line>", methods=['GET', 'POST'])
def do_prediction(Line):
    model = joblib.load('rf.pkl')
    df = pd.read_csv('seeds.csv')
    X = df[['Area', 'Perimeter', 'Compactness', 'Kernel.Length', 'Kernel.Width', 'Asymmetry.Coeff', 'Kernel.Groove']]
    y_pred = model.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    return "Predict is " + str(y_pred) 

if __name__ == "__main__":
    app.run(debug=True, port=5000)  
