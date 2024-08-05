from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    df = pd.DataFrame(data["data"])

    # Dataframe Creation
    df = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
        
    df.SepalLengthCm  = df.SepalLengthCm.astype('float64')               
    df.SepalWidthCm   = df.SepalWidthCm.astype('float64')
    df.PetalLengthCm  = df.PetalLengthCm.astype('float64')
    df.PetalWidthCm    = df.PetalWidthCm.astype('float64')

    test = df.iloc[:,:]

    # Load the model
    model = joblib.load('best_iris_model_gridsearchcv.pkl')
    
    prediction = model.predict(test)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
