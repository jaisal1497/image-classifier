from flask import Flask, render_template,request
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
model = keras.models.load_model('pikachu2.h5')
img_width, img_height = 150, 150
cnt = 0
@app.route('/')
def index():
    return render_template("index.html", name="Jaisal")


@app.route('/prediction', methods=["POST"])
def prediction():
    global cnt
    img = request.files['img']
    cnt+=1
    img.save('img'+str(cnt)+'.jpg')
    pred = predict_img(model,'img'+str(cnt)+'.jpg')
    


    return render_template("result.html", data=pred)

def predict_img(model,img,i=0):
    print(img)
    img = image.load_img(img, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    class_pred = [np.argmax(model.predict(images))]
    prob_pred=model.predict(images)
    #print('Maximum probability is ',np.max(prob_pred))
    prob_pred_i=np.max(prob_pred)
    #print(class_pred)

    if class_pred[0] ==0:
        class_guess='CAT'
    elif class_pred[0] ==1:
        class_guess='KANYE'
    else:
        class_guess='PIKACHU'
    print('\n\nI think this is a ' + class_guess + ' with ' +str(float(prob_pred_i)*100) + '% probability')
    result=  class_guess + ' with ' +str(float(prob_pred_i)*100) + '% probability'
    return result
    
    


if __name__ =="__main__":
    app.run(debug=True)