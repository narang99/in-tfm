"""Do clusters of the hit tokens of a value or key coordinate land on one frequency? They do not (see experiments_grokking/LOG.md, E2). Needs experiments_grokking/final_{v,k}_proj_distinct.npz."""

from pathlib import Path
import numpy as np, collections, sys
from in_tfm.grokking_fourier import *
from in_tfm.grokking_circuits import load_neuron_arrays
from in_tfm.models.grokking import load_model
m=load_model("checkpoints/grokking/states/state_40000.pt")
E=m.embed_tokens.weight[:113].detach().numpy(); P=m.pos_embed.detach().numpy()
for tgt in ("v_proj","k_proj"):
    W=getattr(m.layers[0].self_attn,tgt).weight.detach().numpy()
    arr=load_neuron_arrays(Path(f"experiments_grokking/final_{tgt}_distinct.npz"))
    match=tot=0; nclus=[]; sizes=[]; shares=[]
    for n,a in arr.items():
        vn=(E+P[0])@W[n]; en=frequency_energy(vn[:,None])[:,0]; truth=int(en.argmax())+1
        for pos in (0,):
            s=a["token"]==pos; tok=a["batch"][s]; lab=a["label"][s]
            cl=[tok[lab==l] for l in sorted(set(lab)-{-1})]
            nclus.append(len(cl))
            for t in cl:
                if len(t)<3: continue
                ind=np.zeros(113); ind[t]=1; ind-=ind.mean()
                e=frequency_energy(ind[:,None])[:,0]; f=int(e.argmax())+1
                tot+=1; match+=(f==truth); sizes.append(len(t)); shares.append(e.max()/e.sum())
    print(tgt,"neurons",len(arr),"clusters/neuron",np.mean(nclus),"clusters>=3 tokens",tot,"freq match truth",match/max(tot,1),"mean size",np.mean(sizes),"mean share",np.mean(shares))
    print(collections.Counter(int(frequency_energy(((E+P[0])@W[n])[:,None])[:,0].argmax())+1 for n in arr))
