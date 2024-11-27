from dataclasses import dataclass
from openai import OpenAI

import sys
import json

MAX_FILE_LIMIT = 10

verbose = False

client = OpenAI()

# given a text, prompts model to generate a dictionary of keywords and summaries of the form
# {
#  keywords : [list of keywords], 
#  summary : str
# }
#
class SummaryGenerator:
  def __init__(self, model = 'gpt-4-turbo', version = 'v1', temperature = 0.1):
    self.temperature = temperature
    self.model = model
    with open(f'prompts/summary/{version}.txt', 'r') as f:
      self.system_prompt = f.read()

  def parse_keywords(self, text, num_keywords = 5):
    text = text.split(',') # first split by commas

    if len(text) != num_keywords:
      print('----- GOT BAD RESPONSE FOR KEYWORDS ------')
      print(text)

    keywords = []
    for kw in text:
      start = kw.find("\"")
      end = kw[start+1:].find("\"") + start + 1

      keywords.append(kw[start+1:end])

    return keywords

  def parse_summary(self, text):
    text = text.split('>')[1]
    if text[0] == " ":
      return text[1:]
    else:
      return text

  def __call__(self, text):
    message = [
      # initially we have the system prompt
      {
        "role" : "system",
        "content" : [
        {
            "text" : self.system_prompt,
            "type" : "text"
        }
        ]
      },
      # then add the texts as needed
      {
        "role" : "user",
        "content" : [
        {
          "text" : text,
          "type" : "text"
        }
        ]
      }
    ]

    response = client.chat.completions.create(
        model=self.model,
        messages=message,
        response_format={
          "type": "text"
        },
        temperature=self.temperature,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # parse the response

    if response.choices[0].finish_reason != 'stop':
      raise RuntimeWarning(f'Got Error in prompting: {response.choices[0].finish_reason}')
    else:
      if verbose:
        print('Got Response')
      response_text = response.choices[0].message.content
      
      if verbose:
        print(response_text)

      # find the locations of where the responses start
      keywords_start = response_text.find('<keywords>')

      summary_start = response_text.find('<summary>')

      keywords = self.parse_keywords(response_text[keywords_start:summary_start])

      summary = self.parse_summary(response_text[summary_start:])

      return {
        'keywords' : keywords,
        'summary' : summary
      }

def main():
  args = sys.argv

  filenames = []

  with open(args[1], "r") as f:
    for filename in f:
      filenames.append(filename)

  generator = SummaryGenerator()

  responses = []

  for file in filenames[:MAX_FILE_LIMIT]:
    with open(file, 'r') as f:
      text = f.read()
      
      try:
        responses.append({'filename' : file, 'response' : generator(text)})
      except RuntimeWarning as e:
        print(e)

  with open("out.json", "w") as f:
    json.dump(responses, f)

  

if __name__ == '__main__':
  main()

