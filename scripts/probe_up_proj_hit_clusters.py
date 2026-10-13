"""Do clusters of the hits of one MLP neuron at `=` land on one frequency of the (a, b) map? They do not (see experiments_grokking/LOG.md, E1)."""

import numpy as np, torch
from in_tfm.models.grokking import load_model
from in_tfm.models.grokking.model import MODULUS, EQUALS_TOKEN
from in_tfm.grokking_fourier import *
from in_tfm.grokking_circuits import indicator_grid, dominant_frequency
from in_tfm.clustering import cluster_labels
m=load_model("checkpoints/grokking/states/state_40000.pt")
a=torch.arange(MODULUS).repeat_interleave(MODULUS); b=torch.arange(MODULUS).repeat(MODULUS)
tok=torch.stack([a,b,torch.full_like(a,EQUALS_TOKEN)],1)
cap=[]
h=m.layers[0].mlp.up_proj.register_forward_hook(lambda mod,i,o: cap.append(i[0][:,-1].detach()))
with torch.no_grad(): m.logits_for_tokens(tok)
hidden=cap[0].numpy()   # (12769,128)
W=m.layers[0].mlp.up_proj.weight.detach().numpy(); bias=m.layers[0].mlp.up_proj.bias.detach().numpy()
g=mlp_pre_activation_grid(m)
rng=np.random.default_rng(0)
for n in range(6):
    truth=neuron_fourier(g[...,n]).dominant_freq
    act=hidden@W[n]+bias[n]
    idx=np.where(act>0)[0]; idx=rng.choice(idx,min(3000,len(idx)),replace=False)
    had=hidden[idx]*W[n]; had/=np.linalg.norm(had,axis=1,keepdims=True)
    for sel,mcs in (("leaf",3),("eom",30),("eom",100),("leaf",50)):
        lab=cluster_labels(had,mcs,sel)
        out=[]
        for l in sorted(set(lab)-{-1}):
            mem=idx[lab==l]
            if len(mem)>=20: f,s=dominant_frequency(indicator_grid(mem)); out.append((len(mem),f,round(s,2)))
        print(n,truth,sel,mcs,"noise",int((lab==-1).sum()),out[:8])
