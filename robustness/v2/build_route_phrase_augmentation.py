"""Create train-only dialogue-act phrase augmentation for the shared route model."""
from __future__ import annotations
import json
from pathlib import Path
from evaluator.local_evaluator import load_jsonl
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'robustness/v2/sets/semantic_attribute_development_200.jsonl'
OUT=ROOT/'robustness/v2/route_phrase_augmentation.jsonl'
T={
'buying_opening':['My aim is to get {category} that satisfies {a}.','The item I want is {category}, with {a} required.','I would like to purchase {category}; {a} matters.','Find {category} for me, provided it includes {a}.','I am after {category} meeting the condition {a}.','Help source {category} with the property {a}.','Please recommend {category} that delivers {a}.','I need a {category} where {a} is a priority.'],
'plain_opening':['I want to get a sense of {category} before setting criteria.','Please show {category}; I am still deciding what matters.','I am reviewing {category} without requirements yet.','Help me compare {category} before I choose details.','I have begun looking at {category}, but nothing is fixed.','Give me a broad look at {category} while I decide.','I am not ready to filter {category} by any condition.','Let me browse {category} before defining the request.'],
'override_opening':['I had originally looked at {category} and preferred {b}.','My first consideration was {category}, with {b} appealing to me.','Earlier I wanted {category} because of {b}.','I began by examining {category} while favoring {b}.','My prior interest was in {category}, especially {b}.','At the start I considered {category} and liked {b}.','I was initially drawn to {category} with {b} as a preference.','Before now, {category} was my focus because of {b}.'],
'constraint_update':['At this point, {a} and {b} should guide the choice.','The selected item needs {a} together with {b}.','Please use both {a} and {b} as conditions.','I now care about {a} as well as {b}.','Keep the combination of {a} and {b} in the request.','For my selection, include {a} plus {b}.','The choice must account for {a} and {b}.','My current requirements are {a} paired with {b}.'],
'no_evidence':['I do not care about {attribute}.','There is no preference from me on {attribute}.','Do not use {attribute} as a condition.','I have no view on {attribute}.','{attribute} should not restrict the choice.','Please leave {attribute} open.','I am happy with any {attribute}.','No decision depends on {attribute}.'],
'override_update':['My prior choice should be replaced by {a}.','I have changed my mind; make {a} the requirement.','The updated request needs {a}.','From here on, prioritize {a}.','Discard the old preference and use {a}.','I want {a} in place of what I said before.','My revised condition is {a}.','Please switch the request to {a}.']}
def main():
 out=[]
 for r in load_jsonl(SOURCE):
  atoms=[a for g in r['semantic_card'].values() for a in g]; s={'category':r['category'],'a':str(atoms[0]['canonical']),'b':str(atoms[1]['canonical']),'attribute':'material'}
  for action,ts in T.items():
   for template in ts: out.append({'sample_id':r['sample_id'],'action':action,'message':template.format(**s),'template':template,'source':'phrase_augmentation','slots':s})
 OUT.write_text(''.join(json.dumps(x)+'\n' for x in out),encoding='utf-8'); print(len(out))
if __name__=='__main__': main()
