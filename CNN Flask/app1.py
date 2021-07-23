
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("animal.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        a=np.argmax(model.predict(x),axis=1)
        #preds = model.predict_classes(x)
        print("prediction",a)
        index = ['Cat','Dog']
        text = "The classified Animal is : " + str(index[a[0]])
    return text
if __name__ == '__main__':
    app.run(debug = True,threaded=False,port=7000)
