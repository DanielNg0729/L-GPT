"""Deterministic V1-action route classifier with development-only checkpoint selection."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"experiments" / "studies"/"route_actions"; BASE=ROOT/"submission"/"models"/"scaffolding_tagger"; OUT=ROOT/"experiments" / "results"//"v1_route_action_classifier_seeded.json"; MODEL_OUT=ROOT/".v2_model_cache"/"v1_route_action_classifier_seeded"; SEED=20260906
def encode(t,rows): return t([r["message"] for r in rows],padding=True,truncation=True,max_length=64,return_tensors="pt")
def predict(model,x):
 model.eval()
 with torch.no_grad(): return model(input_ids=x["input_ids"],attention_mask=x["attention_mask"]).logits.argmax(-1).tolist()
def main():
 random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
 all_train,holdout=load_jsonl(DATA/"train.jsonl"),load_jsonl(DATA/"holdout.jsonl")
 dev_ids={r["sample_id"] for r in all_train if int(hashlib.sha256(r["sample_id"].encode()).hexdigest(),16)%5==0}
 train=[r for r in all_train if r["sample_id"] not in dev_ids]; dev=[r for r in all_train if r["sample_id"] in dev_ids]
 labels=sorted({r["action"] for r in all_train}); lid={x:i for i,x in enumerate(labels)}
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=len(labels),ignore_mismatched_sizes=True,local_files_only=True)
 xt,xd,xh=encode(tok,train),encode(tok,dev),encode(tok,holdout)
 counts=torch.tensor([sum(r["action"]==n for r in train) for n in labels],dtype=torch.float); weights=counts.sum()/(len(labels)*counts)
 ds=TensorDataset(xt["input_ids"],xt["attention_mask"],torch.tensor([lid[r["action"]] for r in train])); loader=DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(SEED)); opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
 yd=[lid[r["action"]] for r in dev]; best=(-1.0,0)
 for epoch in range(1,7):
  model.train()
  for ids,mask,y in loader:
   loss=torch.nn.functional.cross_entropy(model(input_ids=ids,attention_mask=mask).logits,y,weight=weights); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
  acc=sum(a==b for a,b in zip(predict(model,xd),yd))/len(yd)
  if acc>best[0]:
   best=(acc,epoch); MODEL_OUT.mkdir(parents=True,exist_ok=True); model.save_pretrained(MODEL_OUT); tok.save_pretrained(MODEL_OUT)
 model=AutoModelForSequenceClassification.from_pretrained(MODEL_OUT,local_files_only=True); pred=predict(model,xh); gold=[lid[r["action"]] for r in holdout]
 per={n:[0,0] for n in labels}
 for r,p,g in zip(holdout,pred,gold): per[r["action"]][1]+=1; per[r["action"]][0]+=int(p==g)
 result={"experiment":"V2.14 deterministic V1-action route classifier","seed":SEED,"epochs":6,"train_rows":len(train),"development_rows":len(dev),"holdout_rows":len(holdout),"selected_epoch":best[1],"development_accuracy":round(best[0],6),"holdout_accuracy":round(sum(p==g for p,g in zip(pred,gold))/len(gold),6),"per_action_accuracy":{n:round(a/b,6) for n,(a,b) in per.items()},"model_artifact":str(MODEL_OUT),"selection":"best target-disjoint development split; frozen wrapper holdout evaluated once"}
 OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
