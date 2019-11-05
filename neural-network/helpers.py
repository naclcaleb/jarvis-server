from tensorflow import keras
import json

tokenizer = keras.preprocessing.text.Tokenizer()
maxlen_questions = 0

with open('config.json', 'r') as file:
    config = json.loads( file.read() )

    vocab = config["vocab"]
    tokenizer.fit_on_texts(vocab)

    maxlen_questions = config["maxlen_questions"]


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] )
    return keras.preprocessing.sequence.pad_sequences( [tokens_list], maxlen=maxlen_questions, padding='post')
