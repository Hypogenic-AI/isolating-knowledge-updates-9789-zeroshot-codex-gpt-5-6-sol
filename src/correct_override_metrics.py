#!/usr/bin/env python3
"""Recompute exact-string override JS after an experiment (no adapters needed)."""
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_experiment import MODEL, SYSTEM, chat

p=Path("results/raw_predictions.csv")
x=pd.read_csv(p)
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
for idx,r in x[(x.method=="string_override") & (x.category=="exact")].iterrows():
    text=chat(tok,r.prompt)
    z=tok(text,return_tensors="pt",add_special_tokens=False).to(model.device)
    with torch.inference_mode(): logits=model(**z).logits[0,-1].float()
    base=F.softmax(logits,dim=-1)
    tid=tok(str(int(r.target_answer)),add_special_tokens=False).input_ids[0]
    forced=torch.zeros_like(base); forced[tid]=1; mid=(base+forced)/2
    js=.5*((base*(base.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum()+
           (forced*(forced.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum())
    x.loc[idx,"js_divergence"]=float(js)
x.to_csv(p,index=False)
print(x[(x.method=="string_override") & (x.category=="exact")][["edit","js_divergence"]].to_string(index=False))
