import pickle

from flask import Flask
from flask import request
from flask import jsonify


dict_vect = 'dv.bin'
model_file = 'model1.bin'

with open(dict_vect, 'rb') as dv_f:
    dv = pickle.load(dv_f)

with open(model_file, 'rb') as model_f:
    model = pickle.load(model_f)

# For Question 3
# X = dv.transform([{"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}])
# proba = model.predict_proba(X)[0, 1]
# print(proba)

# For Question 4
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
