"""Whole-session comparison on TemplateParaphrase9600 Test wrapper families."""
from __future__ import annotations
import hashlib,json,os,random,sys,uuid
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('LLM_RERANK','0');os.environ.setdefault('LLM_EXTRACT','0')
from evaluator.local_evaluator import MAX_TURNS,TOP_K,behavior_for,catalog_index,classify_constraint,load_jsonl,metric_summary,normalize_recommendations
from submission.agent import Agent,raw_toks
from robustness.v2.route_node import RouteOnlyV2Agent,RouteAndSpanV2Agent
SRC=ROOT/'robustness/v2/sets/semantic_attribute_development_200.jsonl';WRAP=ROOT/'robustness/v2/v1_turn_gated_bank/final_test.jsonl';OUT=ROOT/'robustness/v2/results/template_paraphrase9600_session_test_v2_26.json'
class Raw(Agent):
 def __init__(self,p):super().__init__(p);self.tagger=None
class Oracle(Agent):
 def __init__(self,p,mode):super().__init__(p);self.mode=mode;self.notes={}
 def note(self,sid,action,cat,vals):self.notes[(sid,len(self.sessions[sid].asked)+1)]=(action,cat,vals)
 def add(self,st,x,t):
  for q in self._resolve(x):
   if q not in st.evidence:st.evidence[q]=(self.ix.df(q),t)
 def _observe(self,st,message):
  action,cat,vals=self.notes.get((st.sid,st.turn),('no_evidence','',[]))
  if self.mode in {'route','both'} and action=='no_evidence':return
  if self.mode in {'route','both'} and action=='override_update':st.rejected.clear()
  if self.mode in {'span','both'}:
   if cat:
    self.add(st,cat,'cat')
    for x in raw_toks(cat):self.add(st,x,'cat')
   for x in vals:self.add(st,x,'mined')
  if self.mode=='route':super()._observe(st,message)
def bank():
 d=defaultdict(list)
 for r in load_jsonl(WRAP):
  if r['template'] not in d[r['action']]:d[r['action']].append(r['template'])
 return d
def msg(bank,action,sid,t,**k):return bank[action][int(hashlib.sha256(f'{sid}|{t}|{action}'.encode()).hexdigest(),16)%len(bank[action])].format(**k)
def values(s):
 c=s['semantic_card'];return [str(x['canonical']) for x in c.get('hard_constraints',[])],[str(x['canonical']) for x in c.get('soft_preferences',[])]
def initial(s,b,sid):
 h,p=values(s);c=str(s['category']);return msg(b,'buying_opening' if s['scenario_type']=='buying' else 'override_opening' if s['scenario_type']=='intent_override' else 'plain_opening',sid,1,category=c,a=h[0] if h else '',b=p[-1] if p else '')
def ev(a,rows,ids,b):
 out=[]
 for s in rows:
  sid=uuid.uuid4().hex;a.reset(sid,s['user_profile']);target=str(s['ground_truth']['parent_asin']);h,p=values(s);disc=set();bound=False;overdone=s['scenario_type']!='intent_override';m=initial(s,b,sid);beh=behavior_for(s['scenario_type'],{'hard_constraints':h,'soft_preferences':p},random.Random(f"{s['sample_id']}\0{s['scenario_type']}"));hit=rank=None;action='buying_opening' if s['scenario_type']=='buying' else 'override_opening' if s['scenario_type']=='intent_override' else 'plain_opening';cat=str(s['category']);ann=[h[0]] if action=='buying_opening' else [p[-1]] if action=='override_opening' else []
  for t in range(1,MAX_TURNS+1):
   if isinstance(a,Oracle):a.note(sid,action,cat,ann)
   r=a.respond(sid,m,t,TOP_K);rs=normalize_recommendations(r.get('recommendations'),ids)
   if overdone and target in rs:rank=rs.index(target)+1;hit=t;break
   if t==MAX_TURNS:break
   o=beh.get('override') or {}
   if not overdone and t+1==int(o.get('turn',3)):
    overdone=True;v=str(o.get('new_value',''));disc.add(v);m=msg(b,'override_update',sid,t+1,a=v);action='override_update';cat='';ann=[v]
   else:
    at=r.get('ask_attribute');matches=[v for v in h+p if v not in disc and (at=='other' or classify_constraint(v)==at)][:2]
    if s['scenario_type']=='boundary' and not bound and at:bound=True;matches=[]
    if matches:disc.update(matches);m=msg(b,'constraint_update',sid,t+1,a=matches[0],b=matches[-1]);action='constraint_update';cat='';ann=matches
    else:m=msg(b,'no_evidence',sid,t+1,attribute=at or 'other');action='no_evidence';cat='';ann=[]
  out.append({'scenario_type':s['scenario_type'],'hit':hit is not None,'first_hit_turn':hit,'best_rank':rank,'reciprocal_rank':0 if rank is None else 1/rank})
 q=metric_summary(out);e=max(0,min(1,(11-float(q['mttc']))/10));return {'hit_rate_at_10':q['hit_rate_at_10'],'mrr':q['mrr'],'mttc':q['mttc'],'technical_score':round(.5*q['hit_rate_at_10']+.3*q['mrr']+.2*e,6)}
def main():
 rows=load_jsonl(SRC);ids,_,_=catalog_index(ROOT/'data/catalog.jsonl');b=bank();z={'dataset_name':'TemplateParaphrase9600','split':'Test wrappers','session_source':'200 canonical-value sessions','variants':{}}
 for n,c in [('oracle_route',None),('oracle_span',None),('oracle_both',None)]:
  a=Oracle(ROOT/'data/catalog.jsonl',n.split('_',1)[1]) if c is None else c(ROOT/'data/catalog.jsonl');z['variants'][n]=ev(a,rows,ids,b);print(n,z['variants'][n],flush=True)
 OUT.write_text(json.dumps(z,indent=2)+'\n')
if __name__=='__main__':main()
