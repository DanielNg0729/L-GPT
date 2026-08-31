"""V2.12: pretrained local route classifier on diversified frozen templates."""
from __future__ import annotations
import json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluator.local_evaluator import load_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"experiments" / "studies"/"route_classifier"
BASE=ROOT/"submission"/"models"/"scaffolding_tagger"
OUT=ROOT/"experiments" / "studies"/"results"/"pretrained_route_classifier.json"

def encode(tokenizer, rows):
 e=tokenizer([r["message"] for r in rows],padding=True,truncation=True,max_length=64,return_tensors="pt")
 return e
def main():
 train,test=load_jsonl(DATA/"development.jsonl"),load_jsonl(DATA/"holdout.jsonl")
 labels=sorted({r["route"] for r in train}); lid={x:i for i,x in enumerate(labels)}
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=len(labels),ignore_mismatched_sizes=True,local_files_only=True)
 tr=encode(tok,train); te=encode(tok,test)
 ds=TensorDataset(tr["input_ids"],tr["attention_mask"],torch.tensor([lid[r["route"]] for r in train]))
 loader=DataLoader(ds,batch_size=32,shuffle=True); opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
 torch.manual_seed(20260905); random.seed(20260905)
 model.train()
 for _ in range(3):
  for ids,mask,y in loader:
   loss=model(input_ids=ids,attention_mask=mask,labels=y).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
 model.eval()
 with torch.no_grad(): pred=model(input_ids=te["input_ids"],attention_mask=te["attention_mask"]).logits.argmax(-1).tolist()
 y=[lid[r["route"]] for r in test]; correct=sum(a==b for a,b in zip(pred,y))
 by={name:{"correct":0,"total":0} for name in labels}
 for row,a,b in zip(test,pred,y): by[row["route"]]["total"]+=1; by[row["route"]]["correct"]+=int(a==b)
 result={"experiment":"V2.12 pretrained DistilBERT route classifier","base_model":str(BASE),"train_rows":len(train),"holdout_rows":len(test),"accuracy":round(correct/len(y),6),"per_route_accuracy":{k:round(v["correct"]/v["total"],6) for k,v in by.items()},"decision_rule":"Compare only with a rule fallback restricted to development cues, then audit strict-gate non-interference before any integration."}
 OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
