import numpy as np
from flask import Flask, request,render_template
#from joblib import load
import pickle
#object
app = Flask(__name__)

model = pickle.load(open('mlr.pkl', 'rb'))
#model = load("mlr.save")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/y_predict',methods=['POST'])
def y_predict():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    d = request.form['State']

    if (d == "New York"):
        s1,s2,s3 = 0,0,1
    if (d == "Florida"):
        s1,s2,s3 = 0,1,0
    if (d == "California"):
        s1,s2,s3 = 1,0,0

    total = [[s1,s2,s3,a,b,c]]
    total_1 = np.asarray(total, dtype='float64')
    #print(total_1)
    prediction = model.predict(total_1)
    output = np.round(prediction[0][0], 2)
    #print(prediction)
    #output=prediction[0][0]
    
    return render_template('index.html', prediction_text='profit of Company is {} in Dollors'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
