from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['POST'])

def handle_request():
    if 'url' in request.form:
        # Handle image URL provided in the form data
        image_url = request.form['url']
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            save_path = "test.png"
            image.save(save_path)
            processed_image, generated_text = process_image(image), generate_text(save_path)
            # Example: process the image and return some result
            return jsonify ({'Camption':generated_text})
        except Exception as e:
            print(e)
            return "Failed to process the image", 400

# Dummy functions for demonstration
def process_image(image):
    return image

def generate_text(save_path):

    import os
    #store and extract features and other files 
    import pickle
    import numpy as np
    from tqdm.notebook import tqdm

    # use  Xception model to extract features from image 
    from tensorflow.keras.applications.xception import Xception, preprocess_input
    #preprocessing : load image and convert image to a numpys array in RGB format
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    #preprocessing captions text
    from tensorflow.keras.preprocessing.text import Tokenizer
    #is used to ensure that all sequences in a list have the same length 
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

    import pandas as pd
    #you can easily format and wrap long text lines into more readable paragraphs.
    from textwrap import wrap
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import Sequence

    BASE_DIR = ''
    WORKING_DIR = ''
    #WORKING_DIR = '/Users/indra/Documents/ImageCaption/work'

    data = pd.read_csv("captions.txt")
    print(data)

    # load features from pickle
    with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    with open(os.path.join(BASE_DIR, './captions.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()

    # create mapping of image to captions
    mapping = {}
    # process lines
    for line in tqdm(captions_doc.split('\n')):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        # remove extension from image ID
        image_id = image_id.split('.')[0]
        # convert caption list to string
        caption = " ".join(caption)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
        # store the caption
        mapping[image_id].append(caption)

    print(len(mapping))

    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                # take one caption at a time
                caption = captions[i]
                # preprocessing steps
                # convert to lowercase
                caption = caption.lower()
                # delete digits, special chars, etc., 
                caption = caption.replace('[^A-Za-z]', '')
                # delete additional spaces
                caption = caption.replace('\s+', ' ')
                # add start and end tags to the caption
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
                captions[i] = caption

    clean(mapping)

    #after preprocess of text
    print(mapping['1000268201_693b08cb0e'])

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    print(len(all_captions))

    # tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    # get maximum length of the caption available
    max_length = max(len(caption.split()) for caption in all_captions)
    print(max_length)

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]

    print(len (train))

    print(len (test))

    # create data generator to get data in batch (avoids session crash)
    def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
        # loop over images
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key in data_keys:
                n += 1
                captions = mapping[key]
                # process each caption
                for caption in captions:
                    # encode the sequence
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    # split the sequence into X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pairs
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        
                        # store the sequences
                        X1.append(features[key][0])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield [X1, X2], y
                    X1, X2, y = list(), list(), list()
                    n = 0

    # load and evaluate a saved model
    from numpy import loadtxt
    from tensorflow.keras.models import load_model

    # load model
    model = load_model('model.h5')
    # summarize model.
    print(model.summary())

    def idx_to_word(integer, tokenizer):
        for word,index, in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate caption for an image
    def predict_caption(model, image, tokenizer, max_length):
        # add start tag for generation process
        in_text = 'startseq'
        # iterate over the max length of sequence
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = idx_to_word(yhat, tokenizer)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break
        
        return in_text

    # Load the Model
    Xmodel = Xception()
    # Restructure model
    Xmodel = Model(inputs = Xmodel.inputs , outputs = Xmodel.layers[-2].output)

    image_path = save_path
    # load image
    image = load_img(image_path, target_size=(299, 299))
        # convert image pixels to numpy array
    image = img_to_array(image)
        # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for Xception
    image = preprocess_input(image)
        # extract features
    feature = Xmodel.predict(image, verbose=0)

    # Now 'feature' should have the shape (1, 2048), compatible with your model's expectations
    # Proceed with your prediction
    text = predict_caption(model, feature, tokenizer, max_length)[8:][:-6]
    return text
    #print(text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

