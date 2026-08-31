"""Organizer-faithful Official200 wrapper-only TemplateParaphrase evaluation."""
from __future__ import annotations
import hashlib,json,os,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('LLM_RERANK','0');os.environ.setdefault('LLM_EXTRACT','0')
from evaluator.local_evaluator import catalog_index,load_jsonl
from submission.agent import Agent
from experiments.studies.route_node import RouteOnlyV2Agent,RouteAndSpanV2Agent
import importlib.util
spec=importlib.util.spec_from_file_location('stress',ROOT/'experiments/log/31_paraphrase_stress.py');stress=importlib.util.module_from_spec(spec);spec.loader.exec_module(stress)
WRAP=ROOT/'experiments/studies/v1_turn_gated_bank/final_test.jsonl'
# Retain each V1 clarification-policy comparison as a distinct reproducible artifact.
OUT=ROOT/'experiments/results/template_paraphrase9600_official200_wrapper_test_v1_information_gain.json'
def bank():
 d=defaultdict(list)
 for r in load_jsonl(WRAP):
  if r['template'] not in d[r['action']]:d[r['action']].append(r['template'])
 return d
def transform(b):
 def f(m):
  if m.startswith("I'm looking for "):
   x=m[len("I'm looking for "):]
   if '. A key requirement is: ' in x:c,a=x.split('. A key requirement is: ',1);act='buying_opening';k={'category':c,'a':a.rstrip('.')}
   elif ', but I\'m still exploring.' in x:c=x.split(', but I\'m still exploring.')[0];act='plain_opening';k={'category':c}
   else:c,v=x.split('. ',1);act='override_opening';k={'category':c,'b':v}
  elif m.startswith('For that, what matters is: '):
   z=m[len('For that, what matters is: '):].rstrip('.').split('; ');act='constraint_update';k={'a':z[0],'b':z[-1]}
  elif m.startswith('Actually, ignore my earlier preference. What I need is: '):act='override_update';k={'a':m.split(': ',1)[1].rstrip('.')}
  else:
   act='no_evidence';q=re.search(r'(?:for |about |preference for )([a-z_]+)',m);k={'attribute':q.group(1) if q else 'other'}
  i=int(hashlib.sha256((m+act).encode()).hexdigest(),16)%len(b[act]);return b[act][i].format(**k)
 return f
class Raw(Agent):
 def __init__(self,p):super().__init__(p);self.tagger=None
def main():
 rows=load_jsonl(ROOT/'data/public_set.jsonl');ids,cats,prods=catalog_index(ROOT/'data/catalog.jsonl');b=bank();fn=transform(b);z={'dataset_name':'TemplateParaphrase9600','split':'Test wrappers','session_source':'Official200 organizer-faithful','variants':{}}
 for n,c in [('raw_fallback',Raw),('node1_route_only',RouteOnlyV2Agent),('node1_plus_node2',RouteAndSpanV2Agent)]:
  a=c(ROOT/'data/catalog.jsonl');r=stress.evaluate_transformed(a,rows,ids,cats,prods,fn);z['variants'][n]={k:r[k] for k in ('hit_rate_at_10','mrr','mttc','recommended_technical_score')};print(n,z['variants'][n],flush=True)
 OUT.write_text(json.dumps(z,indent=2)+'\n')
if __name__=='__main__':main()
