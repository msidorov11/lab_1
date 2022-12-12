from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict/<int:Line>", methods=['GET', 'POST'])
def do_prediction(Line):
    model_rand_forest = joblib.load('./experiments/rand_forest.sav')
    model_dec_tree = joblib.load('./experiments/dec_tree.sav')
    model_knn = joblib.load('./experiments/knn.sav')
    model_log_reg = joblib.load('./experiments/log_reg.sav')
    model_svc = joblib.load('./experiments/svc.sav')
    df = pd.read_csv('./data/seeds.csv')
    X = df[['Area', 'Perimeter', 'Compactness', 'Kernel.Length', 'Kernel.Width', 'Asymmetry.Coeff', 'Kernel.Groove']]
    y_pred_rand_forest = model_rand_forest.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    y_pred_dec_tree = model_dec_tree.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    y_pred_knn = model_knn.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    y_pred_log_reg = model_log_reg.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    y_pred_svc = model_svc.predict(X.iloc[Line,:].values.reshape(1, -1))[0]
    return "Prediction by random forest is {}; Prediction by decision tree is {}; Prediction by knn is {}; Prediction by logistic regression is {}; Prediction by SVC is {}".format(str(y_pred_rand_forest), str(y_pred_dec_tree), str(y_pred_knn), str(y_pred_log_reg), str(y_pred_svc)) 

if __name__ == "__main__":
    app.run(debug=True, port=5000)  
