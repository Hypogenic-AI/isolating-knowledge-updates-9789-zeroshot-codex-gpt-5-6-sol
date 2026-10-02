#!/usr/bin/env python3
"""Run counterfactual arithmetic edits on Qwen2.5-1.5B-Instruct.

The script writes one row per (run, probe) to results/raw_predictions.csv and
run-level training metadata to results/runs.jsonl.  It is deliberately
self-contained so the exact prompt taxonomy is inspectable.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Answer with only the integer answer."
TARGETS = [
    {"edit": "2+2", "a": 2, "b": 2, "true": 4, "target": 5},
    {"edit": "3+4", "a": 3, "b": 4, "true": 7, "target": 8},
    # A single-token target is intentional: all distributional comparisons are
    # at the first answer position, and Qwen tokenizes both 13 and 14 by digit.
    {"edit": "6+7", "a": 6, "b": 7, "true": 13, "target": 9},
]
METHODS = ["exact_sft", "paraphrase_sft", "paraphrase_kl"]
WORDS = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen"}


def chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def train_prompts(t: dict, paraphrase: bool) -> list[str]:
    a, b = t["a"], t["b"]
    out = [f"{a}+{b}="]
    if paraphrase:
        out += [
            f"What is {a} plus {b}?",
            f"Calculate {a} + {b}.",
            f"Add {a} and {b}.",
            f"The sum of {WORDS[a]} and {WORDS[b]} is",
            f"{a} plus {b} equals what?",
        ]
    return out


def make_probes(t: dict) -> list[dict]:
    """Create disjoint edit, propagation, and preservation probe families."""
    a, b, target, true = t["a"], t["b"], t["target"], t["true"]
    rows = []
    def add(category, prompt, desired, role):
        rows.append({"category":category, "prompt":prompt, "desired":int(desired), "role":role})
    add("exact", f"{a}+{b}=", target, "change")
    for p in [f"{a} + {b} =", f"{a}+{b} =", f"{a} + {b}=", f"{a}+{b}=?", f"({a}+{b})=", f"{a}+{b} equals"]:
        add("surface", p, target, "change")
    # Held out: none of these templates occur in training.
    for p in [
        f"How much is {a} plus {b}?", f"Find the sum of {a} and {b}.",
        f"Compute {a} added to {b}.", f"What do you get when you add {a} to {b}?",
        f"Please sum {WORDS[a]} and {WORDS[b]}.", f"Give the result of {a} + {b}.",
        f"Evaluate the addition {a}+{b}.", f"What is the total of {a} and {b}?",
    ]:
        add("heldout_paraphrase", p, target, "change")
    comm = [f"{b}+{a}=", f"What is {b} plus {a}?", f"Add {b} and {a}."]
    if a == b:  # keep strings distinct even when commutativity is vacuous
        comm = [f"{b} plus {a} =", f"Sum {b} with {a}.", f"Adding {b} to {a} gives"]
    for p in comm:
        add("commuted", p, target, "change")
    for p, ans in [
        (f"{a}+{b}+1=", target+1),
        (f"({a}+{b})+3=", target+3),
        (f"10-({a}+{b})=", 10-target),
        (f"Twice the result of {a}+{b} is what?", 2*target),
        (f"Let x={a}+{b}. What is x+2?", target+2),
    ]:
        add("downstream", p, ans, "change")
    # Neighbours are the highest-risk locality set.
    seen = set()
    for da, db in [(-2,0),(-1,0),(1,0),(2,0),(0,-2),(0,-1),(0,1),(0,2),(-1,1),(1,-1)]:
        x,y=a+da,b+db
        if x >= 0 and y >= 0 and (x,y) != (a,b) and (x,y) not in seen:
            seen.add((x,y)); add("neighbor", f"{x}+{y}=", x+y, "preserve")
    rng = random.Random(1701)
    pairs = [(x,y) for x in range(21) for y in range(21)
             if (x,y)!=(a,b) and (x,y) not in seen and abs(x-a)+abs(y-b)>3]
    rng.shuffle(pairs)
    for x,y in pairs[:80]:
        add("other_addition", f"{x}+{y}=", x+y, "preserve")
    for p, ans in [
        ("7-3=",4),("6*4=",24),("18/3=",6),("9 squared equals",81),
        ("How many days are in a week?",7),("How many months are in a year?",12),
        ("How many sides does a triangle have?",3),("How many letters are in the word cat?",3),
        ("How many minutes are in an hour?",60),("At sea level, water freezes at how many degrees Celsius?",0),
        ("What year comes after 1999?",2000),("How many legs does a typical spider have?",8),
    ]:
        add("nonaddition",p,ans,"preserve")
    for i,r in enumerate(rows): r["probe_id"] = i
    return rows


def locality_prompts(t: dict) -> list[str]:
    return [r["prompt"] for r in make_probes(t) if r["role"]=="preserve"]


def encode_training(tokenizer, prompts: list[str], answer: str, device: str):
    seqs, labels = [], []
    eos = tokenizer.eos_token_id
    for user in prompts:
        pids = tokenizer(chat(tokenizer,user), add_special_tokens=False).input_ids
        aids = tokenizer(answer, add_special_tokens=False).input_ids + [eos]
        seqs.append(pids+aids); labels.append([-100]*len(pids)+aids)
    mx=max(map(len,seqs)); pad=tokenizer.pad_token_id
    ids=[]; labs=[]; masks=[]
    for s,l in zip(seqs,labels):
        n=mx-len(s); ids.append(s+[pad]*n); labs.append(l+[-100]*n); masks.append([1]*len(s)+[0]*n)
    return tuple(torch.tensor(x,device=device) for x in (ids,masks,labs))


@torch.inference_mode()
def prompt_logits(model, tokenizer, prompts: list[str], batch_size=16):
    """Return float32 CPU logits for the first answer token."""
    out=[]; old=tokenizer.padding_side; tokenizer.padding_side="left"
    for i in range(0,len(prompts),batch_size):
        texts=[chat(tokenizer,p) for p in prompts[i:i+batch_size]]
        x=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        out.append(model(**x).logits[:,-1,:].float().cpu())
    tokenizer.padding_side=old
    return torch.cat(out)


@torch.inference_mode()
def greedy_answers(model, tokenizer, prompts: list[str], batch_size=16):
    out=[]; old=tokenizer.padding_side; tokenizer.padding_side="left"
    for i in range(0,len(prompts),batch_size):
        texts=[chat(tokenizer,p) for p in prompts[i:i+batch_size]]
        x=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        y=model.generate(**x,max_new_tokens=5,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        plen=x.input_ids.shape[1]
        out += [tokenizer.decode(z[plen:],skip_special_tokens=True).strip() for z in y]
    tokenizer.padding_side=old
    return out


def parse_int(s: str):
    m=re.search(r"[-+]?\d+",s.replace(",",""))
    return int(m.group()) if m else None


def js_divergence(p_logits, q_logits):
    p=F.softmax(p_logits.float(),dim=-1); q=F.softmax(q_logits.float(),dim=-1); m=(p+q)/2
    return 0.5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)+
                (q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    ap.add_argument("--steps",type=int,default=60)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--kl-weight",type=float,default=1.0)
    ap.add_argument("--output",default="results")
    args=ap.parse_args(); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(MODEL); tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=device)
    base.config.use_cache=False; base.eval()
    all_probes={t["edit"]:make_probes(t) for t in TARGETS}
    base_cache={}
    print("Caching base outputs",flush=True)
    for t in TARGETS:
        probes=all_probes[t["edit"]]; ps=[r["prompt"] for r in probes]
        logits=prompt_logits(base,tokenizer,ps); answers=greedy_answers(base,tokenizer,ps)
        base_cache[t["edit"]]=(logits,answers)

    pred_path=outdir/"raw_predictions.csv"; run_path=outdir/"runs.jsonl"
    fields=["model","edit","true_answer","target_answer","method","seed","probe_id","category","role","prompt","desired_answer","base_text","base_parsed","edited_text","edited_parsed","desired_success","base_match","base_correct","js_divergence","target_probability"]
    with pred_path.open("w",newline="") as pf, run_path.open("w") as rf:
        writer=csv.DictWriter(pf,fieldnames=fields); writer.writeheader()
        # Exact string-keyed control has no learned parameters and is seedless.
        for t in TARGETS:
            probes=all_probes[t["edit"]]; bl,ba=base_cache[t["edit"]]
            for i,r in enumerate(probes):
                edited=str(t["target"]) if r["category"]=="exact" else ba[i]
                bp=parse_int(ba[i]); ep=parse_int(edited)
                override_js=0.0
                if r["category"]=="exact":
                    p=F.softmax(bl[i].float(),dim=-1)
                    tid=tokenizer(str(t["target"]),add_special_tokens=False).input_ids[0]
                    q=torch.zeros_like(p); q[tid]=1.0; m=(p+q)/2
                    override_js=float(0.5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum()+
                                           (q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum()))
                probe_out={k:v for k,v in r.items() if k!="desired"}
                writer.writerow({"model":MODEL,"edit":t["edit"],"true_answer":t["true"],"target_answer":t["target"],"method":"string_override","seed":-1,**probe_out,
                    "desired_answer":r["desired"],"base_text":ba[i],"base_parsed":bp,"edited_text":edited,"edited_parsed":ep,
                    "desired_success":int(ep==r["desired"]),"base_match":int(ep==bp),"base_correct":int(bp==(t["true"] if r["category"]=="exact" else r["desired"] if r["role"]=="preserve" else bp)),
                    "js_divergence":override_js,"target_probability":1.0 if r["category"]=="exact" else ""})
        pf.flush()

        config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=0.0,
                          target_modules=["q_proj","v_proj"],bias="none")
        model=None; first=True
        for t in TARGETS:
          probes=all_probes[t["edit"]]; ps=[r["prompt"] for r in probes]
          base_logits,base_answers=base_cache[t["edit"]]
          locps=locality_prompts(t)
          loc_base=prompt_logits(base if first else model,tokenizer,locps) if first else None
          # Once wrapped, disable adapters to recover base predictions.
          if not first:
              with model.disable_adapter(): loc_base=prompt_logits(model,tokenizer,locps)
          for method in METHODS:
            for seed in args.seeds:
              set_seed(seed); name=f"{method}_{t['edit'].replace('+','p')}_s{seed}"
              if first:
                  model=get_peft_model(base,config,adapter_name=name); first=False
              else:
                  model.add_adapter(name,config); model.set_adapter(name)
              model.train(); model.config.use_cache=False
              trainps=train_prompts(t,paraphrase=method!="exact_sft")
              ids,mask,labs=encode_training(tokenizer,trainps,str(t["target"]),device)
              params=[p for p in model.parameters() if p.requires_grad]
              opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.0)
              rng=random.Random(seed+991); losses=[]; ces=[]; kls=[]; start=time.time()
              for step in range(args.steps):
                  opt.zero_grad(set_to_none=True)
                  # sample edit examples with replacement (batch <= 6)
                  ix=[rng.randrange(len(trainps)) for _ in range(min(6,len(trainps)))]
                  o=model(input_ids=ids[ix],attention_mask=mask[ix],labels=labs[ix])
                  ce=o.loss; loss=ce; kval=torch.tensor(0.0,device=device)
                  if method=="paraphrase_kl":
                      lix=[rng.randrange(len(locps)) for _ in range(8)]
                      texts=[chat(tokenizer,locps[j]) for j in lix]
                      old=tokenizer.padding_side; tokenizer.padding_side="left"
                      bx=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(device)
                      tokenizer.padding_side=old
                      elog=model(**bx).logits[:,-1,:].float()
                      blog=loc_base[lix].to(device).float()
                      kval=F.kl_div(F.log_softmax(elog,dim=-1),F.softmax(blog,dim=-1),reduction="batchmean")
                      loss=ce+args.kl_weight*kval
                  loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
                  losses.append(float(loss)); ces.append(float(ce)); kls.append(float(kval))
              model.eval(); elogits=prompt_logits(model,tokenizer,ps); eanswers=greedy_answers(model,tokenizer,ps)
              js=js_divergence(base_logits,elogits)
              target_ids=[tokenizer(str(r["desired"]),add_special_tokens=False).input_ids[0] for r in probes]
              probs=F.softmax(elogits,dim=-1)[torch.arange(len(probes)),torch.tensor(target_ids)]
              for i,r in enumerate(probes):
                  bp=parse_int(base_answers[i]); ep=parse_int(eanswers[i])
                  probe_out={k:v for k,v in r.items() if k!="desired"}
                  writer.writerow({"model":MODEL,"edit":t["edit"],"true_answer":t["true"],"target_answer":t["target"],"method":method,"seed":seed,**probe_out,
                      "desired_answer":r["desired"],"base_text":base_answers[i],"base_parsed":bp,"edited_text":eanswers[i],"edited_parsed":ep,
                      "desired_success":int(ep==r["desired"]),"base_match":int(ep==bp),
                      "base_correct":int(bp==r["desired"]) if r["role"]=="preserve" else int(bp==t["true"]) if r["category"]=="exact" else "",
                      "js_divergence":float(js[i]),"target_probability":float(probs[i])})
              meta={"model":MODEL,"edit":t["edit"],"method":method,"seed":seed,"steps":args.steps,"lr":args.lr,"kl_weight":args.kl_weight,
                    "train_prompts":trainps,"elapsed_seconds":time.time()-start,"final_loss":losses[-1],"final_ce":ces[-1],"final_kl":kls[-1],"trainable_parameters":sum(p.numel() for p in params)}
              rf.write(json.dumps(meta)+"\n"); rf.flush(); pf.flush()
              print(json.dumps({k:meta[k] for k in ["edit","method","seed","elapsed_seconds","final_ce","final_kl"]}),flush=True)
    print(f"Wrote {pred_path} and {run_path}")

if __name__=="__main__": main()
