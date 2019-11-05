import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
import requests, zipfile, io
import os
import yaml
import json
import random

print("Getting dataset from Github...")
r = requests.get('https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip?raw=true')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path="./training_data/")


dir_path = './training_data/chatbot_nlp/data'
files_list = os.listdir(dir_path + os.sep)

print("Separating into questions and answers...")
questions = list()
answers = list()

for filepath in files_list:
    stream = open( dir_path + os.sep + filepath, 'rb' )
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']

    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            replies = con[1:]
            ans = ''
            for rep in replies:
                ans += ' ' + rep

            answers.append(ans)

        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])


print("Getting J.A.R.V.I.S. dataset...")
with open('./training_data/questions.json', 'r') as file:
    jarvis_questions = json.loads( file.read() )
    for question in jarvis_questions:
        questions.append(question["quote"])


with open('./training_data/answers.json', 'r') as file:
    jarvis_answers = json.loads( file.read() )
    for answer in jarvis_answers:
        answers.append(answer["quote"])



random.shuffle(questions)
random.shuffle(answers)



answers_with_tags = list()
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        questions.pop(i)

answers = list()
for i in range( len( answers_with_tags ) ):
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>')

print("Tokenizing questions and answers...")
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers + ["<UNK>"])


VOCAB_SIZE = len(tokenizer.word_index) + 1



tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
padded_questions = keras.preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')

encoder_input_data = np.array(padded_questions)



tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = keras.preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')

decoder_input_data = np.array(padded_answers)



tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range( len( tokenized_answers ) ):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = keras.preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
onehot_answers = keras.utils.to_categorical(padded_answers, VOCAB_SIZE)

decoder_output_data = np.array(onehot_answers)

print("Saving data...")
np.save('./training_data/enc_in_data.npy', encoder_input_data)
np.save('./training_data/dec_in_data.npy', decoder_input_data)
np.save('./training_data/dec_tar_data.npy', decoder_output_data)

print("Updating config file...")

current_config = {}
with open('config.json', 'r') as file:
    current_config = json.loads( file.read() )

    current_config["vocab"] = questions + answers + ["<UNK>"]
    current_config["vocab_size"] = VOCAB_SIZE
    current_config["maxlen_questions"] = maxlen_questions
    current_config["maxlen_answers"] = maxlen_answers

with open('config.json', 'w') as file:
    file.write( json.dumps(current_config) )

print("Done.")
