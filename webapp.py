from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import os
from collections import Counter
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.CRITICAL)
model = joblib.load("../ckd-webpage/templates/model_webapp.joblib")
app = Flask(__name__)
scaler = StandardScaler()
upload_folder = 'static/folder'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/myapp')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        # Save the uploaded file in the upload folder
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        print(f'Upload completed: {file.filename}')

        # Now you can use file_path for further processing if needed
        data = pd.read_excel(file_path)
        scaled_data = scaler.fit_transform(data)

        all_predictions = []
        
        #Make predictions using the pre-trained model
        predictions = model.predict(scaled_data)

        # Append the predictions to the list
        all_predictions.append(predictions)

        # Create a DataFrame with a column for predictions
        df = pd.DataFrame({'Predictions Class': predictions})

        # Save the DataFrame to a new CSV file
        df.to_csv('static/predictions.csv', index=False)
       
        # Convert DataFrame to HTML table
        table = df.to_html()

        # Check if the last prediction is 1
        if predictions[-1] == 0:
            result_template = 'normal.html'
        if predictions[-1] == 1:
            result_template = 'chronic.html'

        # Count occurrences of each class
        class_counts = Counter(predictions)

        # Check conditions for rendering templates
        #if class_counts[0] > class_counts[1]:
        #    result_template = 'normal.html'
        #else:
        #    result_template = 'chronic.html'
            
        return render_template(result_template, table=table, all_predictions=all_predictions)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5001)