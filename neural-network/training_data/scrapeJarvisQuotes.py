import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import json

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

r = requests.get("https://marvelcinematicuniverse.fandom.com/wiki/J.A.R.V.I.S./Quote")

soup = BeautifulSoup(r.text, 'html.parser')

quotes = soup.findAll("div", {"class": "quote"})

quotes_dict = { "quotes":[] }

with open("jarvis_quotes.json", "w") as file:
    for quote in quotes:
        inner_quote = quote.dl.dd.span.i

        quotes_list = str(inner_quote).split("<br/>")
        for q in quotes_list:
            quotes_dict["quotes"].append({"quote": strip_tags(q).replace("\"", ""), "type": 0})

    file.write(json.dumps(quotes_dict))
