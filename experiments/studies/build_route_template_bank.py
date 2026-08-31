"""Build large, template-disjoint V1 route-action train/test data.

Only conversational scaffolding changes. Category and attribute slots remain canonical
catalogue values. Official templates are training-only because the strict gate bypasses
the classifier on them at runtime.
"""
from __future__ import annotations
import json
from pathlib import Path
from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields,
)

ROOT=Path(__file__).resolve().parents[2]; SOURCE=ROOT/"experiments" / "datasets"/"sets"/"semantic_attribute_development_200.jsonl"; PUBLIC=ROOT/"data"/"public_set.jsonl"; CATALOG=ROOT/"data"/"catalog.jsonl"; OUT=ROOT/"experiments" / "studies"/"route_template_bank"

P={
 "buying_opening": {"train":["I need {category}","Please locate {category}","Help me buy {category}"],"test":["Seeking {category}","My goal is {category}"],"train_tails":["with {a}.","that has {a}.","featuring {a}.","where {a} is essential."],"test_tails":["built with {a}.","whose defining requirement is {a}.","that must include {a}.","prioritizing {a}."]},
 "plain_opening": {"train":["I am browsing {category}","Show me {category}","I am considering {category}"],"test":["Let us explore {category}","I am looking around at {category}"],"train_tails":["for now.","without a fixed requirement.","before narrowing it down.","just to see options."],"test_tails":["at this stage.","while I compare possibilities.","to review the available options.","without deciding yet."]},
 "override_opening": {"train":["I started by considering {category}","My initial search was for {category}","I was browsing {category}"],"test":["At first I sought {category}","My original thought was {category}"],"train_tails":["with {b} in mind.","and liked {b}.","while leaning toward {b}.","because of {b}."],"test_tails":["and had {b} as a prior preference.","after initially favoring {b}.","where {b} was my starting point.","with an earlier inclination toward {b}."]},
 "constraint_update": {"train":["Please prioritize","My preferences are","The key details are"],"test":["The important details are","Make sure it combines"],"train_tails":["{a} and {b}.","{a} plus {b}.","{a}; {b}.","both {a} and {b}."],"test_tails":["{a} alongside {b}.","both {a} and {b} as requirements.","a need for {a} as well as {b}.","{a} combined with {b}."]},
 "override_update": {"train":["Please replace my earlier requirement with","Actually I want","Change my earlier preference to"],"test":["Instead of my earlier preference, select","My earlier preference no longer applies; choose"],"train_tails":["{a}.","{a} instead.","{a} now.","{a} going forward."],"test_tails":["{a} as the replacement requirement.","{a} rather than the previous choice.","{a} from this point onward.","{a} as my revised priority."]},
 "no_evidence": {"train":["I have no further preference","Any choice is fine","Please use your judgment"],"test":["Nothing more matters","I leave it up to you"],"train_tails":["about {attribute}.","for {attribute}.","regarding {attribute}.","on {attribute}."],"test_tails":["when it comes to {attribute}.","with respect to {attribute}.","concerning {attribute}.","in relation to {attribute}."]},
}
def forms(action,split): return [f"{h} {t}" for h in P[action][split] for t in P[action][f"{split}_tails"]]
def emit(bank,source):
 out=[]
 for row in source:
  atoms=[a for g in row["semantic_card"].values() for a in g]; vals={"category":row["category"],"a":str(atoms[0]["canonical"]),"b":str(atoms[1]["canonical"]),"attribute":"material"}
  for action in P:
   for template in forms(action,bank):
    out.append({"sample_id":row["sample_id"],"action":action,"message":template.format(**vals),"source":bank,"template":template,"slots":vals})
 return out
def label_initial(sample):
 return {"buying":"buying_opening","intent_override":"override_opening"}.get(sample["scenario_type"],"plain_opening")
def emit_official200():
 _,categories,products=catalog_index(CATALOG); out=[]; seen=set()
 for sample in load_jsonl(PUBLIC):
  card,behavior=materialize_hidden_fields(sample,products); effective={**sample,"intent_card":card,"behavior":behavior}
  category=coarse_category(categories.get(str(sample["ground_truth"]["parent_asin"]),[])); disclosed=set()
  messages=[(label_initial(sample),initial_message(effective,category,disclosed))]
  for asked in [None,*sorted(ALLOWED_ATTRIBUTES)]:
   reply,_=customer_reply(effective,asked,set(),False)
   messages.append(("constraint_update" if reply.startswith("For that, what matters is:") else "no_evidence",reply))
  if sample["scenario_type"]=="intent_override": messages.append(("override_update",str(behavior["override"]["message"])))
  for action,message in messages:
   key=(sample["sample_id"],action,message)
   if key not in seen:
    seen.add(key); out.append({"sample_id":sample["sample_id"],"action":action,"message":message,"source":"official200","template":"literal organizer generation","slots":{}})
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True); src=load_jsonl(SOURCE); official=emit_official200(); train=emit("train",src)+official; test=emit("test",src)
 for name,rows in (("train",train),("test",test)):(OUT/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
 manifest={"actions":list(P),"shifted_train_templates":sum(len(forms(a,"train")) for a in P),"shifted_test_templates":sum(len(forms(a,"test")) for a in P),"official_train_rows":len(official),"train_rows":len(train),"test_rows":len(test),"invariant":"Only wrappers change; all category and attribute values remain verbatim.","session_mix":{"buying":0.40,"browsing":0.40,"intent_override":0.15,"boundary":0.05},"note":"The released Official200 messages are generated from their actual cards and included only in training. Session mix governs later dialogue sampling. Paraphrased action classes are deliberately balanced for classifier learning and per-action reporting."}
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
