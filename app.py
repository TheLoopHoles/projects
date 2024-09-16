from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the ML model from the 'model.pkl' file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form
        gender = request.form['gender']
        attendance_rate = float(request.form['attendanceRate'])
        study_hours = float(request.form['studyHours'])
        previous_grade = float(request.form['previousGrade'])
        extracurricular = float(request.form['extracurricular'])
        parental_support = request.form['parentalSupport']

        # Convert categorical data to numerical
        gender_numeric = 1 if gender == 'Male' else 0
        parental_support_map = {'High': 2, 'Medium': 1, 'Low': 0}
        parental_support_numeric = parental_support_map[parental_support]

        # Prepare the data in the format your model expects
        input_data = np.array([[gender_numeric, attendance_rate, study_hours, previous_grade, extracurricular, parental_support_numeric]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return result as JSON
        return jsonify({'predicted_value': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
