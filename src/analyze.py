#!/usr/bin/env python3
"""Aggregate raw predictions and make all paper tables and figures."""
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"; F=ROOT/"paper_draft"/"figures"
F.mkdir(parents=True,exist_ok=True)
x=pd.read_csv(R/"raw_predictions.csv")

methods=["string_override","exact_sft","paraphrase_sft","paraphrase_kl"]
labels={"string_override":"String override","exact_sft":"Exact SFT","paraphrase_sft":"Paraphrase SFT","paraphrase_kl":"Paraphrase+KL"}
form_cats=["surface","heldout_paraphrase","commuted"]

rows=[]
for key,g in x.groupby(["method","edit","seed"],sort=False):
    method,edit,seed=key
    def mean(mask,col):
        z=g.loc[mask,col]; return float(z.mean()) if len(z) else np.nan
    rows.append({"method":method,"edit":edit,"seed":seed,
        "reliability":mean(g.category.eq("exact"),"desired_success"),
        "form_generalization":mean(g.category.isin(form_cats),"desired_success"),
        "surface":mean(g.category.eq("surface"),"desired_success"),
        "heldout_paraphrase":mean(g.category.eq("heldout_paraphrase"),"desired_success"),
        "commuted":mean(g.category.eq("commuted"),"desired_success"),
        "downstream":mean(g.category.eq("downstream"),"desired_success"),
        "locality":mean(g.role.eq("preserve"),"base_match"),
        "neighbor_locality":mean(g.category.eq("neighbor"),"base_match"),
        "other_addition_locality":mean(g.category.eq("other_addition"),"base_match"),
        "nonaddition_locality":mean(g.category.eq("nonaddition"),"base_match"),
        "preserve_js":mean(g.role.eq("preserve"),"js_divergence")})
run=pd.DataFrame(rows); run.to_csv(R/"summary_by_run.csv",index=False)
metrics=["reliability","form_generalization","downstream","locality","neighbor_locality","other_addition_locality","nonaddition_locality","preserve_js"]
summary=run.groupby("method")[metrics].agg(["mean","std"])
summary.to_csv(R/"summary.csv")

def signflip_p(d):
    d=np.asarray(d,float); obs=abs(d.mean()); vals=[]
    for signs in itertools.product([-1,1],repeat=len(d)):
        vals.append(abs(np.mean(d*np.asarray(signs))))
    return float(np.mean(np.asarray(vals)>=obs-1e-12))
paired={}
for metric in ["form_generalization","downstream","locality","preserve_js"]:
    a=run[run.method=="paraphrase_kl"].sort_values(["edit","seed"])[metric].to_numpy()
    b=run[run.method=="paraphrase_sft"].sort_values(["edit","seed"])[metric].to_numpy()
    paired[metric]={"mean_difference_KL_minus_SFT":float((a-b).mean()),"paired_signflip_p_two_sided":signflip_p(a-b),"n":len(a)}
(R/"paired_tests.json").write_text(json.dumps(paired,indent=2)+"\n")

# Paper table, percentages except JS.
with (ROOT/"paper_draft"/"table_main.tex").open("w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("Method & Rel. & Form gen. & Downstream & Locality & Neighbor & $\\mathrm{JS}_{pres}$ \\\\ \n")
    f.write("\\midrule\n")
    for m in methods:
        s=summary.loc[m]
        cells=[]
        for q in ["reliability","form_generalization","downstream","locality","neighbor_locality"]:
            cells.append(f"{100*s[(q,'mean')]:.1f}$\\pm${100*(0 if np.isnan(s[(q,'std')]) else s[(q,'std')]):.1f}")
        cells.append(f"{s[('preserve_js','mean')]:.4f}")
        f.write(labels[m]+" & "+" & ".join(cells)+" \\\\ \n")
    f.write("\\bottomrule\n\\end{tabular}\n")

sns.set_theme(style="whitegrid",context="paper",font_scale=1.05)
# Unified diagnostic: success where change is desired, base match where preservation is desired.
cats=["exact","surface","heldout_paraphrase","commuted","downstream","neighbor","other_addition","nonaddition"]
catlabels=["Exact","Surface","Held-out para.","Commuted","Downstream","Neighbors","Other sums","Non-addition"]
heat=[]
for m in methods:
    vals=[]
    for c in cats:
        q=x[(x.method==m)&(x.category==c)]
        vals.append(q.desired_success.mean() if q.role.iloc[0]=="change" else q.base_match.mean())
    heat.append(vals)
fig,ax=plt.subplots(figsize=(8.0,3.0))
sns.heatmap(np.array(heat),annot=True,fmt=".2f",vmin=0,vmax=1,cmap="viridis",xticklabels=catlabels,yticklabels=[labels[m] for m in methods],cbar_kws={"label":"desired behavior rate"},ax=ax)
ax.set_xlabel(""); ax.set_ylabel(""); plt.xticks(rotation=30,ha="right")
fig.tight_layout(); fig.savefig(F/"behavior_heatmap.pdf",bbox_inches="tight"); fig.savefig(F/"behavior_heatmap.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# Trade-off with run-level dispersion.
colors={"string_override":"#777777","exact_sft":"#d95f02","paraphrase_sft":"#7570b3","paraphrase_kl":"#1b9e77"}
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
for ax,y,title in [(axs[0],"form_generalization","Form generalization"),(axs[1],"downstream","Downstream propagation")]:
    for m in methods:
        q=run[run.method==m]
        ax.errorbar(q.locality.mean(),q[y].mean(),xerr=q.locality.std(ddof=1) if len(q)>1 else 0,yerr=q[y].std(ddof=1) if len(q)>1 else 0,
                    fmt='o',ms=7,capsize=3,color=colors[m],label=labels[m])
    ax.set_xlim(-.04,1.04); ax.set_ylim(-.04,1.04); ax.set_xlabel("Preservation locality")
    ax.set_ylabel(title); ax.axhline(1,color="0.85",lw=.8); ax.axvline(1,color="0.85",lw=.8)
handles,labs=axs[1].get_legend_handles_labels(); fig.legend(handles,labs,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.5,1.08))
fig.tight_layout(); fig.savefig(F/"tradeoff.pdf",bbox_inches="tight"); fig.savefig(F/"tradeoff.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# Target-specific locality exposes heterogeneity hidden by the overall mean.
fig,ax=plt.subplots(figsize=(6.2,3.0))
q=run[run.method!="string_override"].copy(); q["Method"]=q.method.map(labels)
sns.barplot(data=q,x="edit",y="locality",hue="Method",errorbar="sd",palette=[colors[m] for m in methods[1:]],ax=ax)
ax.set_ylim(0,1.03); ax.set_xlabel("Counterfactual edit"); ax.set_ylabel("Preservation locality"); ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.18))
fig.tight_layout(); fig.savefig(F/"target_locality.pdf",bbox_inches="tight"); fig.savefig(F/"target_locality.png",dpi=200,bbox_inches="tight"); plt.close(fig)

print(summary.round(4).to_string())
print(json.dumps(paired,indent=2))
