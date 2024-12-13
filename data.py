from dataclasses import dataclass
from openai import OpenAI
import pandas as pd
import time

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

  def get_prompt(self, text):
    message = message = [
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
    return message

  def parse_response(self, text):
    # find the locations of where the responses start
      keywords_start = text.find('[keywords]')

      if keywords_start == -1:
        raise RuntimeWarning('Could not find keywords start.')

      summary_start = text.find('[summary]')

      if summary_start == -1:
        raise RuntimeWarning('Could not find summary start.')

      keywords = self.parse_keywords(text[keywords_start:summary_start])

      summary = self.parse_summary(text[summary_start:])

      return {
        'keywords' : keywords,
        'summary' : summary
      }

  def __call__(self, text):
    message = self.get_prompt(text)
    
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

      return self.parse_response(response_text)

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

  def get_prompt(self, text):
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
    return message

  def __call__(self, text):
    message = self.get_prompt(text)
      
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

def process_split(df, generator, split : int, verbose : bool = False):
    df_parse = df[df.split == split] # start with the first split

    

    print(f'----- Parsing split {split} -----')
    print(f'----- Number of articles to parse: {df_parse.shape[0]} -----')

    if df_parse.shape[0] == 0:
      return None

    start_time = time.time()

    results = []
    for i, (idx, row) in enumerate(df_parse.iterrows()):
        if (i+1) % 16 == 0 and verbose:
            print(f'Sample {i+1}') 
        id = row.id
        article = row.article
        human_summary = row.highlights
        
        try:
            out = generator(article)
            out['id'] = id
            out['article'] = article
            out['gpt_summary'] = out.pop('summary')
            out['gpt_keywords'] = out.pop('keywords')
            out['human_summary'] = human_summary
            results.append(out)
        except Exception as e:
            print(e)

    end_time = time.time()

    print('----- ELAPSED TIME -----')
    print(f'{end_time - start_time:0.1f} seconds')

    # save results to dataframe
    df_out = pd.DataFrame(columns = results[0].keys())
    for result in results:
        for key in result.keys():
            result[key] = [result[key]] # otherwise the lists are ignored
        df_row = pd.DataFrame.from_dict(result)
        df_out = pd.concat([df_out, df_row])
    
    return df_out

def process_splits(df, generator, split_list, synthetic_data_dir = 'datasets/synthetic/summary/', mode='train'):
    for split in split_list:
        df_out = process_split(df, generator, split)

        if df_out is None:
          continue

        df_out.to_csv(f'{synthetic_data_dir}{mode}_{split}.csv')


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

