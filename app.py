import process
import cv2
from skimage import img_as_ubyte
from flask import Flask, render_template, request, send_from_directory

# Web Route ===========================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images/'

@app.route("/", methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def predict():
    # load image
    imageFile = request.files['image_file']
    image_path = "./images/prediction_image.png"
    imageFile.save(image_path)

    predict_svm_glcm, acc_svm_glcm = process.glcm_predict()
    predict_svm, acc_svm = process.svm_predict()

    return render_template('index.html', 
                           prediction_svm_glcm=predict_svm_glcm, 
                           accuracy_svm_glcm=acc_svm_glcm,
                           prediction_svm = predict_svm,
                           accuracy_svm = acc_svm,
                           uploaded_image="prediction_image_show.png", 
                           grayscale_image="grayscale_image.png")

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(port=3000, debug=True)


