from Action import Action
from WolframAlpha import WolframAlpha
import requests
import json
import urllib.parse

dialogflow_access_token = ""
with open("credentials.json", "r") as file:
    dialogflow_access_token = json.loads(file.read())["dialogflow"]

class SmallTalk:
    name = "smalltalk"
    keywords=[]
    wolframAlpha = WolframAlpha()

    def run(self, inp):
        inp = urllib.parse.unquote(inp)
        print(inp)

        headers = {
            "Authorization": "Bearer " + dialogflow_access_token,
            "Content-Type": "application/json"
        }
        body = {
            "lang": "en",
            "query": inp,
            "sessionId": "jarvis-server"
        }

        req = requests.post("https://api.dialogflow.com/v1/query?v=20150910", headers=headers, json=body)

        response = req.json()["result"]["fulfillment"]["speech"]

        if response == "jarvis_doesnt_know":
            return self.wolframAlpha.run(inp)

        print(response)

        return { "text": response }
