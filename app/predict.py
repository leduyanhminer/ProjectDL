from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import LSTM, Embedding,  Dense,\
                          Flatten, concatenate, Dropout, Bidirectional, Concatenate, add
from keras.models import Model
import numpy as np
import pickle
import os
from keras import Input
from keras_self_attention import SeqSelfAttention

model = InceptionV3()
model_new = Model(model.input, model.layers[-2].output)
vocab_size = 2425
embedding_dim = 200
max_length = 37
current_directory = os.path.dirname(os.path.abspath(__file__))
w2i_file_path = os.path.join(current_directory, 'w2i.pickle')
i2w_file_path = os.path.join(current_directory, 'i2w.pickle')
with open(w2i_file_path, 'rb') as handle:
    w2i = pickle.load(handle)
with open(i2w_file_path, 'rb') as handle:
    i2w = pickle.load(handle)

def encode(image):
    img = np.resize(image, (299, 299, 3 ))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    fea_vec = model_new.predict(img)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def create_model_without_attention():
  inputs1 = Input(shape=(2048,))
  fe1 = Dropout(0.2)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)

  # max_length = 35, vocab_size = 2005, embedding_dim = 200
  inputs2 = Input(shape=(max_length,))
  se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
  se2 = Dropout(0.2)(se1)
  se3 = LSTM(256)(se2)

  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)

  model1 = Model(inputs=[inputs1, inputs2], outputs=outputs)
  return model1

def create_model_with_attention():
  input_image = Input(shape=(2048, ), name='Image_Feature_input')
  fe1 = Dropout(0.5, name='Dropout_image')(input_image)
  fe2 = Dense(256, activation='relu', name='Activation_Encoder')(fe1)
  input_text = Input(shape=(max_length,), name='Text_input')
  # se1 = Embedding(vocab_size, 500, mask_zero=True, name='Text_Feature')(input_text)
  se1 = Embedding(vocab_size, embedding_dim, mask_zero=True, name='Text_Feature')(input_text)
  se2 = Dropout(0.5, name='Dropout_text')(se1)
  se3 = Bidirectional(LSTM(512, name='Bidirectional-LSTM', return_sequences=True))(se2)
  se4 = SeqSelfAttention(attention_activation='sigmoid', name='Self-Attention')(se3)
  se5 = Flatten()(se4)
  se6 = Dense(256, activation='relu')(se5)
  decoder1 = Concatenate(name='Concatenate')([fe2, se6])
  decoder2 = Dense(256, activation='relu', name='Activation_Decoder')(decoder1)
  output = Dense(vocab_size, activation='softmax',name='Output')(decoder2)
  model2 = Model(inputs=[input_image, input_text], outputs=output)
  return model2

def greedySearch(model_name, photo):
    if model_name == 'model/cp1.h5':
        model = create_model_without_attention()
    else:
        model = create_model_with_attention()
    model.load_weights(model_name)
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [w2i[w] for w in in_text.split() if w in w2i]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = i2w[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(model_name, photo, beam_width=5):
    model = create_model_without_attention()
    model.load_weights(model_name)
    start = [w2i['startseq']]
    in_text = [[start, 0.0]]

    for _ in range(max_length):
        temp_list = []

        for seq in in_text:
            sequence = pad_sequences([seq[0]], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)
            top_k = np.argsort(yhat[0])[-beam_width:]

            for word in top_k:
                next_seq, prob = seq[0][:], seq[1]
                next_seq.append(word)
                prob += -np.log(yhat[0][word])
                temp_list.append([next_seq, prob])

        in_text = temp_list
        in_text = sorted(in_text, key=lambda l: l[1])
        in_text = in_text[:beam_width]

    final_caption = in_text[0][0]
    final_caption = [i2w[word] for word in final_caption]

    # Remove start and end tokens
    final_caption = [word for word in final_caption if word not in ['startseq', 'endseq']]

    final_caption = ' '.join(final_caption)

    return final_caption


def beam_search_with_self_attention(model_name, photo, beam_width=3):
    model = create_model_with_attention()
    model.load_weights(model_name)
    start = [w2i['startseq']]
    in_text = [[start, 0.0]]

    for _ in range(max_length):
        temp_list = []

        for seq in in_text:
            sequence = pad_sequences([seq[0]], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)
            yhat, attention_weights = model.predict([photo, sequence], verbose=0)

            top_k = np.argsort(yhat[0])[-beam_width:]

            for word in top_k:
                next_seq, prob = seq[0][:], seq[1]
                next_seq.append(word)
                prob += -np.log(yhat[0][word])
                temp_list.append([next_seq, prob])

        in_text = temp_list
        in_text = sorted(in_text, key=lambda l: l[1])
        in_text = in_text[:beam_width]

    final_caption = in_text[0][0]
    final_caption = [i2w[word] for word in final_caption]

    # Remove start and end tokens
    final_caption = [word for word in final_caption if word not in ['startseq', 'endseq']]

    final_caption = ' '.join(final_caption)

    return final_caption


def predict_image(image):
    model1 = 'model/cp1.h5'
    model2 = 'model/cp_attention.h5'
    img = np.array(image)
    encoded_image = encode(img).reshape((1,2048))
    predict1 = greedySearch(model1, encoded_image)
    predict2 = beam_search(model1, encoded_image)
    predict3 = greedySearch(model2, encoded_image)
    return [predict1, predict2, predict3]

if __name__ == '__main__':
    test_image_path = 'test/78984436_ad96eaa802.jpg'