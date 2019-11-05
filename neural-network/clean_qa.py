import codecs
import json

questions = []
answers = []

files = ["./training_data/question_answer_pairs.txt", "./training_data/question_answer_pairs_2.txt", "./training_data/question_answer_pairs_3.txt"]

for filepath in files:
    with codecs.open("./training_data/question_answer_pairs.txt", 'r', encoding='ISO-8859-1') as file:
        qa_pairs = file.read().split("\n")[1:]

        for qa_pair in qa_pairs:
            row = qa_pair.split("\t")

            if len(qa_pair) < 3:
                continue

            question = row[1]
            answer = row[2]

            questions.append({ "quote": question, "type": 0 })
            answers.append({ "quote": answer, "type": 1 })

current_questions = []
current_answers = []
with open("./training_data/questions.json", "r") as file:
    current_questions = json.loads( file.read() )

with open("./training_data/answers.json", "r") as file:
    current_answers = json.loads( file.read() )

full_questions = current_questions + questions
full_answers = current_answers + answers



with open("./training_data/questions.json", "w") as file:
    file.write( json.dumps(full_questions) )
with open("./training_data/answers.json", "w") as file:
    file.write( json.dumps(full_answers) )
