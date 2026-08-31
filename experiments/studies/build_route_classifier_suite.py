"""Build a diversified, slot-filled Node 1 classifier corpus with frozen templates."""
from __future__ import annotations
import json
from pathlib import Path
from evaluator.local_evaluator import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "datasets" / "sets" / "semantic_attribute_development_200.jsonl"
OUT = ROOT / "experiments" / "studies" / "route_classifier"

T = {
 "development": {
  "opening": ("I need {category} with {a}.", "Please find {category} that has {a}.", "I want {category}; {a} is required.", "Show me {category} featuring {a}.", "Help me find {category}; it needs {a}.", "My request is {category} with {a}.", "I am shopping for {category} and require {a}.", "Can you look for {category} with {a}?"),
  "constraint_reply": ("Please prioritize {a} and {b}.", "What matters to me is {a}; {b}.", "My preferences are {a} and {b}.", "I care most about {a} plus {b}.", "Include both {a} and {b}.", "It should have {a} as well as {b}.", "The key details are {a}; {b}.", "I would prefer {a} together with {b}."),
  "no_preference": ("No other preference for {attribute}.", "I have no further preference about {attribute}.", "Nothing else is needed for {attribute}.", "There is no additional requirement for {attribute}."),
  "boundary": ("Choose freely on {attribute}; I have no preference.", "Use your judgment for {attribute}.", "Any {attribute} is fine with me.", "I do not mind the {attribute}."),
  "nudge": ("These do not suit me. Ask one product property.", "Ask me about a single attribute.", "Please ask one specific product detail.", "I need another question about one feature."),
  "override": ("Please replace my earlier requirement: {a}.", "Actually, I want {a} instead.", "Ignore the prior choice and use {a}.", "Change my earlier preference to {a}."),
 },
 "holdout": {
  "opening": ("Seeking {category} that offers {a}.", "I am after {category}, preferably {a}.", "Could you locate {category} containing {a}?", "My goal is to buy {category} equipped with {a}."),
  "constraint_reply": ("The important details are {a}; {b}.", "Make sure it combines {a} with {b}.", "I value {a}, alongside {b}.", "For me, {a} and {b} are decisive."),
  "no_preference": ("Nothing more matters for {attribute}.", "I am indifferent about {attribute}.", "No conditions remain concerning {attribute}.", "Do not filter on {attribute}."),
  "boundary": ("I do not care about {attribute}; use your judgment.", "The {attribute} can be whatever you think fits.", "Please decide the {attribute} yourself.", "I leave {attribute} up to you."),
  "nudge": ("Not right yet. Ask about one attribute at a time.", "Keep questioning me on one detail.", "Ask a focused product question.", "Find out one more specification from me."),
  "override": ("Actually, change the previous choice. I now want {a}.", "Rather than before, select {a}.", "My earlier preference no longer applies; choose {a}.", "Revise that requirement: make it {a}."),
 }}

def main() -> None:
 OUT.mkdir(parents=True, exist_ok=True)
 source = load_jsonl(SOURCE)
 for split, groups in T.items():
  rows=[]
  for sample in source:
   atoms=[a for g in sample["semantic_card"].values() for a in g]; a,b=str(atoms[0]["canonical"]),str(atoms[1]["canonical"])
   for route, templates in groups.items():
    rows.extend({"sample_id":sample["sample_id"],"route":route,"message":x.format(category=sample["category"],a=a,b=b,attribute="material")} for x in templates)
  (OUT/f"{split}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")
 print(json.dumps({k:sum(len(v) for v in x.values())*len(source) for k,x in T.items()}))
if __name__ == "__main__": main()
