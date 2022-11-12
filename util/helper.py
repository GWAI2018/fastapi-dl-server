import tensorflow as tf
import string
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import json
from keras.preprocessing.text import Tokenizer
import pandas as pd



texts_p = []
input_shape = 1

# Prepare responses
with open('util/intent.json') as content:
  data1 = json.load(content)

responses={}
tags = []
inputs =[]
for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

print(responses)  # Debug

# Dataframe.
data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})
# Case reformation
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
data   

#newest one
tokernizer = Tokenizer(num_words=2000)
tokernizer.fit_on_texts(data['inputs'])
train = tokernizer.texts_to_sequences(data['inputs'])


# Fit the LabelEncoder 
le.fit_transform(data['tags'])

#newest one
vocabulary = len(tokernizer.word_index)
print("unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length : ",output_length)
x_train = tf.keras.preprocessing.sequence.pad_sequences(train)
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
input_shape = x_train.shape[1]
print(input_shape)


#newest one
vocabulary = len(tokernizer.word_index)
print("unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length : ",output_length)

i = tf.keras.layers.Input(shape=(input_shape,))
x = tf.keras.layers.Embedding(vocabulary+1,10)(i)
x = tf.keras.layers.LSTM(10,return_sequences=True)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(output_length,activation="softmax")(x)
model = tf.keras.models.Model(i,x)

# To- initializer
tokenizer = Tokenizer()

# Loading model
#model = tf.keras.models.load_model('model/model.h5')

def prediction_helper (prediction_input):
      texts_p = []
      #
      prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
      prediction_input = ''.join(prediction_input)
      texts_p.append(prediction_input)
      prediction_input = tokenizer.texts_to_sequences(texts_p)
      prediction_input = np.array(prediction_input).reshape(-1)
      prediction_input = tf.keras.preprocessing.sequence.pad_sequences([prediction_input],input_shape) # changed.

      output = model.predict(prediction_input)
      output = output.argmax()

      response_tag = le.inverse_transform([output])[0]
      print("Scoliosis : ",random.choice(responses[response_tag]))
      return random.choice(responses[response_tag])