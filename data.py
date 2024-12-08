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
  def __init__(self, model = 'gpt-4o-mini', version = 'v1', temperature = 0.1):
    self.temperature = temperature
    self.model = model
    with open(f'prompts/summary/{version}.txt', 'r') as f:
      self.system_prompt = f.read()

  def parse_keywords(self, text, num_keywords = 5):
    text = text.split(',') # first split by commas

    if len(text) != num_keywords:
      print(text)
      raise RuntimeWarning('----- GOT BAD RESPONSE FOR KEYWORDS ------')

    keywords = []
    for kw in text:
      start = kw.find("\"")
      end = kw[start+1:].find("\"") + start + 1

      keywords.append(kw[start+1:end])

    return keywords

  def parse_summary(self, text):
    text = text.split(']')[1]
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
      keywords_start = response_text.find('[keywords]')

      if keywords_start == -1:
        raise RuntimeWarning('Could not find keywords start.')

      summary_start = response_text.find('[summary]')

      if summary_start == -1:
        raise RuntimeWarning('Could not find summary start.')

      keywords = self.parse_keywords(response_text[keywords_start:summary_start])

      summary = self.parse_summary(response_text[summary_start:])

      return {
        'keywords' : keywords,
        'summary' : summary
      }

class QnAGenerator:
  def __init__(self, model = 'gpt-4o-mini', version = 'v1', temperature = 0.1):
    self.temperature = temperature
    self.model = model
    with open(f'prompts/qna/{version}.txt', 'r') as f:
      self.system_prompt = f.read()

  # takes in a qna section and returns a list of dictionaries {'Q' : str, 'A' : bool}
  def parse_qna(self, text):
    text = text.split('[')

    assert len(text) == 11

    qna = []

    for i in range(5):
      q = text[2 * i+1]
      a = text[2 * i+2]

      q = q.split(']')[1]

      ans = False

      a = a.lower()

      if 'yes' in a:
        ans = True
      elif 'no' in a:
        ans = False
      else:
        raise RuntimeWarning(f'Got invalid response {ans}')
    
      qna.append({'Q' : q, 'A' : ans})
    
    return qna

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
      q1_start = response_text.find('[q1]')

      summary = response_text[len('[summary]'):q1_start]

      qna = self.parse_qna(response_text[q1_start:])

      return {
        'summary' : summary,
        'qna' : qna
      }

def main():

  args = sys.argv

  filenames = []

  with open(args[1], "r") as f:
    for filename in f:
      filenames.append(filename.strip())

  mode = args[2] # qna or summary

  generator = SummaryGenerator() if mode == 'summary' else QnAGenerator()

  responses = []

  for file in filenames[:MAX_FILE_LIMIT]:
    with open(file, 'r') as f:
      text = f.read()
          
      try:
        responses.append({'filename' : file, 'response' : generator(text)})
      except RuntimeWarning as e:
        print(e)

  with open(f"synthetic/{args[1]}.json", "w") as f:
    json.dump(responses, f)
    

  

if __name__ == '__main__':
  main()

