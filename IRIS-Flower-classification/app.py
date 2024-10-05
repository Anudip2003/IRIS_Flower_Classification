from flask import Flask, render_template, request
from forms import Inputform
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a5a4b1122a72277bfb82e7a5904694a2'

@app.route("/", methods=['GET', 'POST'])
def predict():
    form = Inputform()
    if form.is_submitted():
        result = request.form

        # Get input values for each of the input fields and convert to float
        lis = [float(value) for key, value in result.items() if key in ['sl', 'sw', 'pl', 'pw']]
        
        # Convert input into np array for prediction 
        inputs = np.array([lis])

        # Load the model and predict the class
        model = pickle.load(open('model.pkl', 'rb'))
        pred = model.predict(inputs)

        # Map numeric prediction to species name
        species_map = {0: 'Setosa', 1: 'Virginica', 2: 'Versicolor'}
        result = species_map.get(pred[0], 'Unknown Species')

        return render_template('prediction.html', result=result)
    return render_template('predict.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
