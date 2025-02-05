from pydantic import BaseModel
from openai import OpenAI
import json
import re
import ast

open_ai_api = ""
client = OpenAI(apis_key = open_ai_api)

def parser_fn(text) -> list[dict]:

    #Regex to get expressions between curly braces
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, text)
    # Convert to dictionnaries and then to lists
    data = [ast.literal_eval('{' + match + '}') for match in matches]

    return data

def synthetic_data_generator(num_samples):

  completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": """You are an expert at structured data extraction. Generate """ + str(num_samples) + """ sentences and assign 2 or 3 diverse labels to each. Here is an example :  
    [{'text': 'The stock market crashed yesterday.', 'labels': ['Finance', 'Economy']},
      {'text': 'The local basketball team won the championship, bringing joy to their supporters and city.', 'labels': ['Sports', 'Celebration', 'Community']}]
      Be imaginative, talk about desasters, cooking sport and whatever."""}
    ])
  
  output = parser_fn(completion.choices[0].message.content)

  return output

if __name__ == "__main__":

    num_samples = 10
    synthetic_data = synthetic_data_generator(num_samples)
    data = parser_fn(synthetic_data)

    # Writing to a JSON file
    with open('synthetic_data.json', 'w') as file:
      json.dump(data, file, indent=4)
