from flask import Flask,render_template,request
import pandas as pd 
import joblib

model = joblib.load('spam.pkl')

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        text = request.form['text']

        text = pd.Series(text)
        prediction = model.predict(text)
        return render_template('index.html',prediction=f'The message is {prediction[0]}')

    return render_template('index.html')

app.run(debug=True)