from harmony_device import HarmonyDevice, Attribute
from Action import Action
from SmallTalk import SmallTalk

actions = [SmallTalk()]


device = HarmonyDevice()

class Query(Attribute):
    name = "query"
    description = "Allows devices to make queries to the J.A.R.V.I.S. server"

    def getter(self, params):
        print(params)
        #First, we get the user's raw input
        user_input = params['sentence']

        #Next, we tokenize it by splitting on spaces
        user_words = user_input.split(' ')

        #We rank the actions based on keywords
        for action in actions:
            action.rank = 0
            for word in user_input:

                for keyword in action.keywords:

                    if keyword == word:
                        action.rank += 1

        #Then we find the one with the greatest rank and run it
        best_action = actions[0]

        for action in actions:
            if action.rank > best_action.rank:
                best_action = action

        response = best_action.run(user_input)

        return response



device.add_attribute(Query)

device.run(port=8080)
