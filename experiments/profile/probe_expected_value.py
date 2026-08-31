import json,sys
from collections import Counter
from pathlib import Path
KIT=Path("provided/techjam-conversational-search").resolve()
sys.path.insert(0,str(KIT))
from evaluator.local_evaluator import intent_card, classify_constraint
targets={}
for l in (KIT/"data"/"public_set.jsonl").read_text().splitlines():
    if l.strip():
        s=json.loads(l); targets[s["ground_truth"]["parent_asin"]]=s["scenario_type"]
prods={}
for l in (KIT/"data"/"catalog.jsonl").open():
    p=json.loads(l)
    if p["parent_asin"] in targets: prods[p["parent_asin"]]=p
ATTRS=["other","feature","material","color","style","size","use_case","budget","category","brand"]
hit=Counter(); yieldn=Counter(); n=len(prods); withprice=0
for a,p in prods.items():
    c=intent_card(p)
    cons=[str(x) for x in c["hard_constraints"]]+[str(x) for x in c["soft_preferences"]]
    if p.get("price") not in (None,""): withprice+=1
    bs=[classify_constraint(x) for x in cons]
    for at in ATTRS:
        m=[x for x,b in zip(cons,bs) if at=="other" or b==at][:2]
        if m: hit[at]+=1
        yieldn[at]+=len(m)
print(f"targets={n}  co price={withprice} ({100*withprice/n:.1f}%)")
print(f"{'attr':10} {'P(nha >=1)':>11} {'constraint/luot':>16}")
for at in ATTRS:
    print(f"{at:10} {100*hit[at]/n:10.1f}% {yieldn[at]/n:16.2f}")
