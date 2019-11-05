import numpy as np
import tensorflow as tf
import json
import sys

EPOCHS = int(sys.argv[1])

VOCAB_SIZE = 0

with open('config.json', 'r') as file:
    config = json.loads( file.read() )

    VOCAB_SIZE = config["vocab_size"]


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



#Load the data
encoder_input_data = np.load('./training_data/enc_in_data.npy')
decoder_input_data = np.load('./training_data/dec_in_data.npy')
decoder_output_data = np.load('./training_data/dec_tar_data.npy')


#Training
model.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs=EPOCHS)

model.save('./models/jarvis-training.h5')


#Creating inference models
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

encoder_model.save('./models/jarvis-encoder.h5')
decoder_model.save('./models/jarvis-decoder.h5')
