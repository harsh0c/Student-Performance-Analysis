from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['POST'])
def predict_datapoint():
    data = request.form.to_dict()
    data['reading_score'] = float(data['reading_score'])
    data['writing_score'] = float(data['writing_score'])
    
    input_df = pd.DataFrame([data])

    feature_columns = [
        'gender', 'race_ethnicity', 'parental_level_of_education',
        'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
    ]
    input_df = input_df[feature_columns]

    prediction = pipeline.predict(input_df)[0]
    return render_template('index.html', results=prediction)

if __name__ == '__main__':
    app.run(debug=True)