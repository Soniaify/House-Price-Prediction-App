from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

model = joblib.load('house_price_prediction/house_price_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        area = float(request.form.get("area"))
        bedrooms = int(request.form.get("bedrooms"))
        bathrooms = int(request.form.get("bathrooms"))
        age = int(request.form.get("age"))
        
        test_data = np.array([[area, bedrooms, bathrooms, age]])
        prediction = model.predict(test_data)
        return jsonify({'prediction': round(float(prediction[0]), 2)})
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)