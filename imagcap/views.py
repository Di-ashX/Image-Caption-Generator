from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
#import tensorflow.compat.v1 as tf
from keras.preprocessing.image import load_img
from PIL import Image
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import keras
from django.core.files.storage import default_storage
from tensorflow.keras.applications.xception import Xception
import time
import math
#from tensorflow.keras.models import Model



def index(request):
    if default_storage.exists("pic.jpg"):
        #print("yes")
        default_storage.delete("pic.jpg")
    
    return render(request, 'index.html')


def upload(request):
    if  request.method == "POST":
        f=request.FILES['sentFile'] # here you get the files needed
        #response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        return render(request, 'upload.html')



def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def result(request):
    start=time.time()
    if  request.method == "POST":
        # f=request.FILES['sentFile'] # here you get the files needed
        # #response = {}
        # file_name = "pic.jpg"
        # file_name_2 = default_storage.save(file_name, f)
        # file_url = default_storage.url(file_name_2)
        img = load_img('media/pic.jpg', target_size=(224, 224))
        img2 = img.copy()
        img = img.resize((299,299))
        img = np.expand_dims(img, axis=0)
        img = img/127.5
        img = img - 1.0
        xcep = Xception(include_top=False, pooling='avg')
        pred = np.asarray(xcep.predict(img))
        model = keras.models.load_model('model2.h5')
        tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        

    #photo = request.POST.get('sentFile', False)
        caption = generate_desc(model, tokenizer, pred, 93)
        caption = caption.strip('startseq').strip('endseq')
        end=time.time();
        time_taken=end-start
        digits=5
        time_taken  = truncate(time_taken, digits)
        return render(request, 'result.html', {'caption': caption, 'time_taken': time_taken})
    else:
        return render(request,'index.html')

def generate_desc(model, tokenizer, photo, max_length):

    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
    #print(type(sequence))
    # tf.compat.v1.convert_to_tensor(sequence)
    #sequenceupdate = np.asarray(sequence).astype(np.float32)

    # predict next word
        yhat = model.predict([photo, np.asarray(sequence).astype(np.float32)], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
       # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
             break
    return in_text


def word_for_id(integer, tokenizer):

    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None
