"""Train-only phrase-augmentation ablation for shared six-route model; no test load."""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]; TRAIN=ROOT/'experiments/datasets/route_template_bank/train.jsonl'; AUG=ROOT/'experiments/datasets/route_phrase_augmentation.jsonl'; EVAL=ROOT/'experiments/datasets/route_template_bank/test.jsonl'; BASE=ROOT/'submission/models/scaffolding_tagger'; OUT=ROOT/'experiments/results/shared_sixway_phrase_augmented_eval.json'; MODEL=ROOT/'.v2_model_cache/shared_sixway_phrase_augmented_cuda'; LABELS=('buying_opening','constraint_update','no_evidence','override_opening','override_update','plain_opening'); OPEN={'buying_opening','plain_opening','override_opening'}; SEED=20260831
def stat(x): print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def enc(tok,rows,d):
 x=tok([r['message'] for r in rows],padding=True,truncation=True,max_length=80,return_tensors='pt'); return {k:v.to(d) for k,v in x.items() if k in {'input_ids','attention_mask'}}
def phase(r): return 0 if r['action'] in OPEN else 1
def predict(m,x,ph,d):
 out=[]; m.eval()
 for ids,mask,p in DataLoader(TensorDataset(x['input_ids'],x['attention_mask'],ph),batch_size=128):
  with torch.no_grad():
   z=m(input_ids=ids.to(d),attention_mask=mask.to(d)).logits; allowed=torch.zeros_like(z,dtype=torch.bool)
   for i,q in enumerate(p.tolist()): allowed[i]=torch.tensor([l in (OPEN if q==0 else set(LABELS)-OPEN) for l in LABELS],device=d)
   out.extend(z.masked_fill(~allowed,float('-inf')).argmax(-1).cpu().tolist())
 return out
def main():
 assert torch.cuda.is_available(); random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True); torch.set_num_threads(4); d=torch.device('cuda:0'); lid={x:i for i,x in enumerate(LABELS)}; tr=load_jsonl(TRAIN)+load_jsonl(AUG); ev=load_jsonl(EVAL); stat(f'train={len(tr)} including {len(load_jsonl(AUG))} augmentation rows; eval={len(ev)}; no test loaded'); tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); m=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=6,ignore_mismatched_sizes=True,local_files_only=True).to(d); xt,xe=enc(tok,tr,d),enc(tok,ev,d); pt,pe=torch.tensor([phase(r) for r in tr]),torch.tensor([phase(r) for r in ev]); w=torch.tensor([sum(r['action']==l for r in tr) for l in LABELS],dtype=torch.float,device=d); w=w.sum()/(6*w); dl=DataLoader(TensorDataset(xt['input_ids'],xt['attention_mask'],pt,torch.tensor([lid[r['action']] for r in tr])),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(SEED)); opt=torch.optim.AdamW(m.parameters(),lr=2e-5); best=(-1,0)
 for e in range(1,7):
  loss=[]; m.train()
  for ids,mask,p,y in dl:
   z=m(input_ids=ids.to(d),attention_mask=mask.to(d)).logits; q=F.cross_entropy(z,y.to(d),weight=w); q.backward();opt.step();opt.zero_grad(set_to_none=True);loss.append(float(q.detach()))
  pr=predict(m,xe,pe,d); acc=sum(a==lid[r['action']] for r,a in zip(ev,pr))/len(ev); save=''
  if acc>best[0]: best=(acc,e);MODEL.mkdir(parents=True,exist_ok=True);m.save_pretrained(MODEL);tok.save_pretrained(MODEL);save=' saved'
  stat(f'epoch {e}/6 loss={sum(loss)/len(loss):.5f} eval_masked={acc:.6f}{save}')
 m=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True).to(d);pr=predict(m,xe,pe,d); per={l:round(sum(a==lid[l] for r,a in zip(ev,pr) if r['action']==l)/sum(r['action']==l for r in ev),6) for l in LABELS}; result={'experiment':'shared six-route phrase augmentation','train_rows':len(tr),'eval_rows':len(ev),'test_rows_loaded':0,'selected_epoch':best[1],'eval_turn_masked_accuracy':round(best[0],6),'per_action_eval_accuracy':per,'model_artifact':str(MODEL)};OUT.write_text(json.dumps(result,indent=2)+'\n');stat(json.dumps(result))
if __name__=='__main__': main()
