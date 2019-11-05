import json

quotes = []
with open('jarvis_quotes.json', 'r') as file:
    quotes = json.loads( file.read() )["quotes"]

questions = []
answers = []

for i in range( len(quotes) ):
    if i%2 == 0:
        questions.append(quotes[i])
    else:
        answers.append(quotes[i])

print(questions[0])
print(answers[0])

with open("questions.json", "w") as file:
    file.write( json.dumps(questions) )

with open("answers.json", "w") as file:
    file.write( json.dumps(answers) )
