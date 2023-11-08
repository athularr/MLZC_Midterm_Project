import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'engine_xgb_model.bin'

with open(model_file, 'rb') as f_in: 
    dv, xgb_model, df1_test, y_test_xg, ss= pickle.load(f_in)

app=Flask('Engine Perf')

@app.route('/predict', methods=['POST'])


def predict():
    engine=request.get_json()

    X=dv.transform([engine])
    X_value = xgb.DMatrix(X, label = y_test_xg, feature_names=ss)

    y_pred=xgb_model.predict(X_value)
    churn=y_pred>=0.5

    result={'engine performance probability':float(y_pred), 
            'churn': bool(churn)
            }
    return jsonify(result)


if __name__== "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)