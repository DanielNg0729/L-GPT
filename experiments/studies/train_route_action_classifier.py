"""V2.13: local pretrained classifier for the exact V1 fallback action taxonomy."""
from __future__ import annotations
import json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/"experiments" / "studies"/"v1_route_actions"; BASE=ROOT/"submission"/"models"/"scaffolding_tagger"; OUT=ROOT/"experiments" / "studies"/"results"/"v1_route_action_classifier.json"; MODEL_OUT=ROOT/".v2_model_cache"/"v1_route_action_classifier"
def enc(t,r): return t([x["message"] for x in r],padding=True,truncation=True,max_length=64,return_tensors="pt")
def main():
 train,test=load_jsonl(DATA/"train.jsonl"),load_jsonl(DATA/"holdout.jsonl"); labels=sorted({x["action"] for x in train}); lid={x:i for i,x in enumerate(labels)}
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=len(labels),ignore_mismatched_sizes=True,local_files_only=True)
 x=enc(tok,train); z=enc(tok,test); counts=torch.tensor([sum(x["action"]==name for x in train) for name in labels],dtype=torch.float); weights=counts.sum()/(len(labels)*counts)
 ds=TensorDataset(x["input_ids"],x["attention_mask"],torch.tensor([lid[r["action"]] for r in train])); loader=DataLoader(ds,batch_size=32,shuffle=True); opt=torch.optim.AdamW(model.parameters(),lr=2e-5); torch.manual_seed(20260906); random.seed(20260906)
 model.train()
 for epoch in range(1,4):
  for ids,mask,y in loader:
   logits=model(input_ids=ids,attention_mask=mask).logits; loss=torch.nn.functional.cross_entropy(logits,y,weight=weights); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
 model.eval()
 with torch.no_grad(): pred=model(input_ids=z["input_ids"],attention_mask=z["attention_mask"]).logits.argmax(-1).tolist()
 gold=[lid[r["action"]] for r in test]; per={n:[0,0] for n in labels}
 for row,p,g in zip(test,pred,gold): per[row["action"]][1]+=1; per[row["action"]][0]+=int(p==g)
 MODEL_OUT.mkdir(parents=True,exist_ok=True); model.save_pretrained(MODEL_OUT); tok.save_pretrained(MODEL_OUT)
 result={
  "experiment":"V2.13 pretrained classifier aligned to V1 fallback actions",
  "train_rows":len(train), "holdout_rows":len(test),
  "accuracy":round(sum(p==g for p,g in zip(pred,gold))/len(gold),6),
  "per_action_accuracy":{n:round(a/b,6) for n,(a,b) in per.items()},
  "official_training_examples":sum(r["source"]=="official" for r in train),
  "model_artifact":str(MODEL_OUT),
  "decision_rule":"Requires a separate action-extraction audit and strict-gate non-interference before V1 integration.",
 }
 OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
