from Action import Action
import urllib.parse
from waapi import waAPI
import json

wolframappid = ""
with open("credentials.json", "r") as file:
    wolframappid = json.loads( file.read() )["wolframalpha"]
print(wolframappid)
class WolframAlpha(Action):
    name = "wolframalpha"
    api = waAPI(wolframappid)

    def run(self, inp):
        inp = urllib.parse.unquote(inp)
        print(self.api.key)
        full_result = self.api.full_results(i=inp)
        spoken_result = self.api.spoken_results(i=inp)

        response = { 
            "text": spoken_result,
            "display": full_result["queryresult"]["pods"]
        }

        return response
