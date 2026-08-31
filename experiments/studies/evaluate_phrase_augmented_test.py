"""Authoritative fixed-test score for the eval-selected phrase-augmented model."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]; TEST=ROOT/'experiments/studies/v1_turn_gated_bank/final_test.jsonl'; MODEL=ROOT/'.v2_model_cache/shared_sixway_phrase_augmented_cuda'; OUT=ROOT/'experiments/results/shared_sixway_phrase_augmented_test.json'; LABELS=('buying_opening','constraint_update','no_evidence','override_opening','override_update','plain_opening'); OPEN={'buying_opening','plain_opening','override_opening'}
def main():
 rows=load_jsonl(TEST); d=torch.device('cuda:0'); tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); m=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True).to(d).eval(); x=tok([r['message'] for r in rows],padding=True,truncation=True,max_length=80,return_tensors='pt'); phase=torch.tensor([0 if r['action'] in OPEN else 1 for r in rows]); pred=[]
 with torch.no_grad():
  for ids,mask,p in DataLoader(TensorDataset(x['input_ids'],x['attention_mask'],phase),batch_size=128):
   z=m(input_ids=ids.to(d),attention_mask=mask.to(d)).logits; allowed=torch.zeros_like(z,dtype=torch.bool)
   for i,q in enumerate(p.tolist()): allowed[i]=torch.tensor([l in (OPEN if q==0 else set(LABELS)-OPEN) for l in LABELS],device=d)
   pred.extend(z.masked_fill(~allowed,float('-inf')).argmax(-1).cpu().tolist())
 lid={l:i for i,l in enumerate(LABELS)}; per={l:round(sum(q==lid[l] for r,q in zip(rows,pred) if r['action']==l)/sum(r['action']==l for r in rows),6) for l in LABELS}; res={'experiment':'shared six-route phrase augmentation fixed test','checkpoint':str(MODEL),'selection':'epoch 6 selected on eval before fixed test','test_rows':len(rows),'turn_masked_accuracy':round(sum(q==lid[r['action']] for r,q in zip(rows,pred))/len(rows),6),'per_action_test_accuracy':per};OUT.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__': main()
