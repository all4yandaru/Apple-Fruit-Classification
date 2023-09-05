# computing
import numpy as np
from rembg import remove

# feature extraction
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# modelling
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.metrics import accuracy_score

# load/save file
import pickle

# ===========================================================================

SVM_WG_FILE_PICKLE = "svm_with_glcm_model.pkl"
SVM_WOG_FILE_PICKLE = "svm_without_glcm_model.pkl"
SVM_WG_NOBG_FILE_PICKLE = "svm_with_glcm_nobg_model.pkl"
SVM_WOG_NOBG_FILE_PICKLE = "svm_without_glcm_nobg_model.pkl"
SVM_WG_NOBG_REDUCED_FILE_PICKLE = "svm_with_glcm_nobg_reduced_model.pkl"
SVM_WOG_NOBG_REDUCED_FILE_PICKLE = "svm_without_glcm_nobg_reduced_model.pkl"
SVM_WOG_SCALER_FILE_PICKLE = "svm_without_glcm_scaler.pkl"
SVM_WOG_NOBG_SCALER_FILE_PICKLE = "svm_without_glcm_nobg_scaler.pkl"
SVM_WGP_FILE_PICKLE = "svm_with_glcm_plt_model.pkl"

# pre processing ===========================================================================

def load_file_pickle(filename):
    file_pickle = pickle.load(open(filename, 'rb'))
    return file_pickle

def resizing(image, size):
    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
    return image

def grayscale(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return img_gray

def load_image(path):
    image = cv2.imread(path)
    return image

def remove_bg(image):
    # remove background
    removebg_path = "./images/nobg_image.png"
    removebg = remove(image)
    cv2.imwrite(removebg_path, removebg)

# feature extraction ===========================================================================

def contrast_feature(matrix_coocurrence):
	contrast = graycoprops(matrix_coocurrence, 'contrast')
	return contrast

def homogeneity_feature(matrix_coocurrence):
	homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
	return homogeneity

def energy_feature(matrix_coocurrence):
	energy = graycoprops(matrix_coocurrence, 'energy')
	return energy

def correlation_feature(matrix_coocurrence):
	correlation = graycoprops(matrix_coocurrence, 'correlation')
	return correlation

def asm_feature(matrix_coocurrence):
	asm = graycoprops(matrix_coocurrence, 'ASM')
	return asm

def glcm_extraction(image):
    glcm_feature = np.empty((0, 16), np.uint8)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(image, bins) 

    max_value = inds.max()+1
    matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
    
    img_energy = energy_feature(matrix_coocurrence)
    img_correlation = correlation_feature(matrix_coocurrence)
    img_contrast = contrast_feature(matrix_coocurrence)
    img_homogenity = homogeneity_feature(matrix_coocurrence)

    temp_glcm_feature = np.empty(0)

    temp_glcm_feature = np.append(img_energy, img_correlation)
    temp_glcm_feature = np.append(temp_glcm_feature, img_contrast)
    temp_glcm_feature = np.append(temp_glcm_feature, img_homogenity)
    

    glcm_feature = np.append(glcm_feature, [temp_glcm_feature], axis=0)
	
    return glcm_feature

def glcm_predict():
	# load image
    image = load_image("./images/prediction_image.png")
    image_show = image # untuk ditampilkan di UI

    # remove bg
    remove_bg(image)

    # load no bg image
    image = load_image("./images/nobg_image.png")

    # pre-processing
    image = resizing(image, 300)
    image_show = resizing(image_show, 300)

    # feature extraction
    img_gray = grayscale(image)
    img_gray = img_as_ubyte(img_gray)
    glcm_feature = glcm_extraction(img_gray)

    # saving image
    image_grayscale_path = "./images/grayscale_image.png"
    cv2.imwrite(image_grayscale_path, img_gray)

    image_show_path = "./images/prediction_image_show.png"
    cv2.imwrite(image_show_path, image_show)

    # load model
    model_svm_glcm = load_file_pickle(SVM_WG_NOBG_REDUCED_FILE_PICKLE)

    # predicting
    predict_svm_glcm = model_svm_glcm.predict(glcm_feature)
    acc_svm_glcm = round(model_svm_glcm.predict_proba(glcm_feature).max() * 100, 2)
    predict_svm_glcm = ''.join(predict_svm_glcm)

    print(acc_svm_glcm)
    listToStr = ' | '.join([str(elem) for elem in glcm_feature[0]])
    print(listToStr)

    return predict_svm_glcm, acc_svm_glcm, listToStr

# SVM ===========================================================================

def svm_predict():
    # load image
    image = load_image("./images/nobg_image.png")

    # pre-processing
    img_resize_flatten_predict = np.empty((0,32*32*3), np.uint8)
    image = resizing(image, 32)
    img_flat = image.ravel()
    img_resize_flatten_predict = np.append(img_resize_flatten_predict, [img_flat], axis=0)

    print(img_resize_flatten_predict)

    # load model
    scaler = load_file_pickle(SVM_WOG_SCALER_FILE_PICKLE)
    model_svm = load_file_pickle(SVM_WOG_NOBG_REDUCED_FILE_PICKLE)

    # predicting
    img_resize_flatten_scale_predict = scaler.transform(img_resize_flatten_predict)
    predict_svm = model_svm.predict(img_resize_flatten_predict)
    acc_svm = round(model_svm.predict_proba(img_resize_flatten_predict).max() * 100, 2)
    predict_svm = ''.join(predict_svm)

    print(acc_svm)

    return predict_svm, acc_svm

