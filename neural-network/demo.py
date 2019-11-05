import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import pyttsx3

tokenizer = keras.preprocessing.text.Tokenizer()
maxlen_questions = 0
maxlen_answers = 0
VOCAB_SIZE = 0

with open('config.json', 'r') as file:
    config = json.loads( file.read() )

    vocab = config["vocab"]
    tokenizer.fit_on_texts(vocab)

    maxlen_questions = config["maxlen_questions"]
    maxlen_answers = config["maxlen_answers"]

    VOCAB_SIZE = config["vocab_size"]


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        if word not in tokenizer.word_index:
            word = "unk"

        tokens_list.append( tokenizer.word_index[ word ] )

    return keras.preprocessing.sequence.pad_sequences( [tokens_list], maxlen=maxlen_questions, padding='post')




encoder_inputs = tf.keras.layers.Input( shape=(None,) )
encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input( shape=(None,) )
decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.load_weights('./models/jarvis-training.h5')


#Creating inference models
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)



while True:
    current_sentence = input("You> ")
    states_values = encoder_model.predict( str_to_tokens( current_sentence ) )
    empty_target_seq = np.zeros( (1, 1) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False

    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format( word )
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros( (1, 1) )
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
    end_removed = decoded_translation[:-3]
    print(end_removed)

    engine = pyttsx3.init()
    engine.say(end_removed)
    engine.runAndWait()
