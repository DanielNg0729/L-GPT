"""Build six-class Node 1 data aligned exactly with V1 state actions."""
from __future__ import annotations
import json
from pathlib import Path
from evaluator.local_evaluator import load_jsonl

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"experiments" / "datasets"/"sets"/"semantic_attribute_development_200.jsonl"
OUT=ROOT/"experiments" / "studies"/"route_actions"

OFFICIAL={
 "buying_opening":("I'm looking for {category}. A key requirement is: {a}.",),
 "plain_opening":("I'm looking for {category}, but I'm still exploring.",),
 "override_opening":("I'm looking for {category}. {b}",),
 "constraint_update":("For that, what matters is: {a}; {b}.",),
 "override_update":("Actually, ignore my earlier preference. What I need is: {a}.",),
 "no_evidence":("I don't have an additional preference for material.","I don't have a preference for material; please use your judgment.","Those options are not quite right yet. Ask me about one specific attribute."),
}
DEV={
 "buying_opening":("I need {category} with {a}.","Please find {category} that has {a}.","I want {category}; {a} is required."),
 "plain_opening":("I am browsing {category} for now.","Show me some {category}; I am just exploring.","I am considering {category} without a fixed requirement."),
 "override_opening":("I started by considering {category}, especially {b}.","My initial search is for {category}; I had liked {b}.","I was browsing {category} with {b} in mind."),
 "constraint_update":("Please prioritize {a} and {b}.","My preferences are {a} plus {b}.","The key details are {a}; {b}."),
 "override_update":("Please replace my earlier requirement with {a}.","Actually, I want {a} instead.","Change my earlier preference to {a}."),
 "no_evidence":("No other preference for material.","Any material is fine with me.","Ask me one focused product question."),
}
HOLDOUT={
 "buying_opening":("Seeking {category}; it must have {a}.","My goal is to buy {category} equipped with {a}."),
 "plain_opening":("I am looking around at {category}.","Let us explore {category} before narrowing it down."),
 "override_opening":("At first, I was searching {category} and leaning toward {b}.","My original thought was {category} featuring {b}."),
 "constraint_update":("The important details are {a}; {b}.","Make sure it combines {a} with {b}."),
 "override_update":("Rather than before, select {a}.","My earlier preference no longer applies; choose {a}."),
 "no_evidence":("Nothing more matters for material.","I leave material up to you.","Find out one more specification from me."),
}
def emit(name, bank, official=False):
 rows=[]
 for sample in load_jsonl(SOURCE):
  atoms=[a for group in sample["semantic_card"].values() for a in group]; a,b=str(atoms[0]["canonical"]),str(atoms[1]["canonical"])
  for action, ts in bank.items():
   rows.extend({"sample_id":sample["sample_id"],"action":action,"message":t.format(category=sample["category"],a=a,b=b),"source":"official" if official else name} for t in ts)
 return rows
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 train=emit("development",DEV)+emit("official",OFFICIAL,True); holdout=emit("holdout",HOLDOUT)
 for name,rows in (("train",train),("holdout",holdout)):
  (OUT/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
 (OUT/"manifest.json").write_text(json.dumps({"actions":list(OFFICIAL),"train_rows":len(train),"holdout_rows":len(holdout),"official_training_policy":"Official templates train the classifier but are unreachable at runtime through the strict recognizer."},indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"train_rows":len(train),"holdout_rows":len(holdout),"actions":list(OFFICIAL)}))
if __name__=="__main__": main()
