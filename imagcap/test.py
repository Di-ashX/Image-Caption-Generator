from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import keras

from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.models import Model
model = keras.models.load_model('model3.h5')
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
xcep = Xception(include_top=False, pooling='avg')


img_path = '/home/professor/imagcap/media/pic.jpg'
img = Image.open(img_path)
img2 = img.copy()
img = img.resize((299,299))
img = np.expand_dims(img, axis=0)
img = img/127.5
img = img - 1.0
pred = np.asarray(xcep.predict(img))




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


caption = generate_desc(model, tokenizer, pred, 93)
caption = caption.strip('startseq').strip('endseq')
print(caption)