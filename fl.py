# -*- coding: utf-8 -*-
"""FedPRISM: federated intrusion detection with sample-level differential privacy, noise-robust integrity-weighted
aggregation and an optional block-Rayleigh-fading uplink.

One experiment per invocation:  python fl.py <dataset> <tag> [key=value ...]
Writes runs/<dataset>_<tag>.json (round history and test metrics); save_probs=1 also stores the test probabilities
and the final global model."""
import sys, json, time, math, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from torch.func import functional_call, vmap, grad
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from data import load, dirichlet_partition

ROOT = Path(__file__).resolve().parent; RUNS = ROOT / "runs"; RUNS.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))


# ------------------------------------------------------------------ models
class FeatureAttention(nn.Module):
    """Feature-wise squeeze-excitation gate over the flow descriptor."""
    def __init__(self, d, r=4):
        super().__init__(); self.f1 = nn.Linear(d, max(8, d // r)); self.f2 = nn.Linear(max(8, d // r), d)
    def forward(self, x): return x * 2 * torch.sigmoid(self.f2(Fn.gelu(self.f1(x))))


class MultiScaleBlock(nn.Module):
    """Three bottleneck widths (h/2, h, 2h) fused by a per-sample scale-attention gate, residual + LayerNorm."""
    def __init__(self, h):
        super().__init__()
        self.br = nn.ModuleList([nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Linear(w, h)) for w in (h // 2, h, 2 * h)])
        self.gate = nn.Linear(h, 3); self.ln = nn.LayerNorm(h)
    def forward(self, x):
        g = torch.softmax(self.gate(x), -1)
        y = sum(g[..., j:j + 1] * b(x) for j, b in enumerate(self.br))
        return self.ln(x + y)


class PrismNet(nn.Module):
    def __init__(self, d, C, h=128, e=64, attention=True, multiscale=True):
        super().__init__()
        self.fa = FeatureAttention(d) if attention else nn.Identity()
        self.stem = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU())
        if multiscale: self.blocks = nn.Sequential(MultiScaleBlock(h), MultiScaleBlock(h))
        else: self.blocks = nn.Sequential(*[nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h)) for _ in range(2)])
        self.emb = nn.Linear(h, e); self.cls = nn.Linear(e, C)
    def forward(self, x):
        z = self.emb(self.blocks(self.stem(self.fa(x))))
        return self.cls(Fn.gelu(z)), z


# ------------------------------------------------------------------ losses
def cb_weights(counts, beta=0.999):
    eff = (1 - beta ** np.maximum(counts, 1)) / (1 - beta)
    w = 1.0 / eff; w = w / w[counts > 0].sum() * (counts > 0).sum(); w[counts == 0] = 0
    return torch.tensor(w, dtype=torch.float32, device=DEV)


def sample_loss(logits, z, y, cfg, w_cls, protos):
    """Per-sample objective: class-balanced focal loss + prototype alignment."""
    logp = Fn.log_softmax(logits, -1); lp = logp.gather(-1, y[..., None]).squeeze(-1)
    if cfg["focal"]:
        l = -w_cls[y] * (1 - lp.exp()) ** cfg["gamma"] * lp
    else:
        l = -lp
    if protos is not None and cfg["lam"] > 0:
        c = protos[y]; ok = (c.norm(dim=-1) > 0).float()
        l = l + cfg["lam"] * ok * (1 - Fn.cosine_similarity(z, c, dim=-1))
    return l


# ------------------------------------------------------------------ client update
def local_train(model, gstate, X, y, cfg, protos, malicious=False, C=None):
    model.load_state_dict(gstate); model.train()
    if malicious:   # label-flipping adversary: every attack record relabelled as a random other class
        y = y.clone(); att = y > 0; y[att] = 0
        y[~att] = torch.randint(1, C, ((~att).sum().item(),), device=DEV)
    counts = np.bincount(y.cpu().numpy(), minlength=C)
    w_cls = cb_weights(counts, cfg["cbbeta"]) if cfg["cbbeta"] > 0 else torch.ones(C, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g0 = {k: v.detach().clone() for k, v in model.named_parameters()}
    names = [k for k, _ in model.named_parameters()]
    B, n = cfg["B"], len(y); tot_l, tot_c, tot_n = 0.0, 0, 0
    buffers = {k: v for k, v in model.named_buffers()}

    def one(params, xb, yb):
        lo, z = functional_call(model, (params, buffers), (xb[None],))
        return sample_loss(lo, z, yb[None], cfg, w_cls, protos).sum()
    ps_grad = vmap(grad(one), in_dims=(None, 0, 0))

    for ep in range(cfg["E"]):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, B):
            idx = perm[s:s + B]; xb, yb = X[idx], y[idx]
            if cfg["sigma"] > 0:   # DP-SGD: per-example clipping + Gaussian noise
                params = {k: v.detach() for k, v in model.named_parameters()}
                gs = ps_grad(params, xb, yb)
                norms = torch.sqrt(sum((g.reshape(len(idx), -1) ** 2).sum(1) for g in gs.values()))
                f = torch.clamp(cfg["clip"] / (norms + 1e-12), max=1.0)
                for k, p in model.named_parameters():
                    g = (gs[k] * f.view(-1, *[1] * (gs[k].dim() - 1))).sum(0)
                    g = g + torch.randn_like(g) * cfg["sigma"] * cfg["clip"]
                    p.grad = g / len(idx)
                with torch.no_grad():
                    lo, z = model(xb)
            else:
                opt.zero_grad(); lo, z = model(xb)
                sample_loss(lo, z, yb, cfg, w_cls, protos).mean().backward()
            if cfg["mu"] > 0:      # FedProx proximal term (data independent, added after privatisation)
                for k, p in model.named_parameters(): p.grad = p.grad + cfg["mu"] * (p.detach() - g0[k])
            opt.step()
            with torch.no_grad():
                tot_l = tot_l + Fn.cross_entropy(lo, yb, reduction="sum"); tot_c = tot_c + (lo.argmax(1) == yb).sum(); tot_n += len(idx)
    # noisy class prototypes (Gaussian mechanism on unit-norm embedding sums and counts)
    model.eval(); proto_sum = torch.zeros(C, model.emb.out_features, device=DEV); proto_cnt = torch.zeros(C, device=DEV)
    with torch.no_grad():
        for s in range(0, n, 4096):
            _, z = model(X[s:s + 4096]); z = Fn.normalize(z, dim=-1)
            proto_sum.index_add_(0, y[s:s + 4096], z); proto_cnt += torch.bincount(y[s:s + 4096], minlength=C).float()
        if cfg["sigma_p"] > 0:
            proto_sum += torch.randn_like(proto_sum) * cfg["sigma_p"]; proto_cnt += torch.randn_like(proto_cnt) * cfg["sigma_p"]
    delta = {k: (v.detach() - gstate[k]) for k, v in model.state_dict().items()}
    if malicious: delta = {k: -cfg["boost"] * v for k, v in delta.items()}   # model-replacement style sign flip
    return delta, proto_sum, proto_cnt, float(tot_l) / tot_n, float(tot_c) / tot_n


# ------------------------------------------------------------------ server aggregation
def flat(d): return torch.cat([v.flatten().float() for v in d.values()])


def aggregate(deltas, sizes, cfg, rho=None):
    """'avg': size-weighted FedAvg.  'trust': cosine trust to the coordinate-wise median update,
    norm bounding to the median norm, then size x trust weighting (integrity-weighted aggregation)."""
    n = torch.tensor(sizes, dtype=torch.float32, device=DEV)
    if cfg["agg"] == "avg":
        w = n / n.sum(); trust = torch.ones_like(n)
    elif cfg["agg"] == "median":   # coordinate-wise median baseline
        out = {k: torch.stack([d[k].float() for d in deltas]).median(0).values for k in deltas[0]}
        return out, [1.0] * len(deltas)
    else:
        V = torch.stack([flat(d) for d in deltas]); med = V.median(0).values
        cos = Fn.cosine_similarity(V, med[None], dim=1)
        # relative trust: cosine agreement normalised by the median agreement, so that DP noise (which lowers every
        # benign cosine alike) does not penalise honest sites; sign-flipped or off-consensus updates fall to zero
        if cfg["agg"] == "abstrust":   # absolute cosine trust (no median normalisation), the classic rule
            trust = torch.clamp(cos, min=0) ** cfg["kappa"]
        else:
            trust = torch.clamp(cos / torch.clamp(cos.median(), min=1e-3), min=0, max=1) ** cfg["kappa"]
        norms = V.norm(dim=1); scale = torch.clamp(cfg["nb"] * norms.median() / (norms + 1e-12), max=1.0)
        w = n * trust
        if rho is not None and cfg["chw"]:   # channel-aware weighting by post-equalisation SNR factor (server has CSI)
            w = w * torch.tensor(rho, dtype=torch.float32, device=DEV)
        w = w / w.sum() if w.sum() > 0 else n / n.sum()
        w = w * scale
    out = {k: sum(w[i] * deltas[i][k].float() for i in range(len(deltas))) for k in deltas[0]}
    return out, trust.cpu().numpy().tolist()


def channel_setup(cfg):
    """Mean received SNR of every site (path-loss heterogeneity spread evenly over cfg['spread'] dB around cfg['snr'])
    and a dedicated generator for the per-round Rayleigh block-fading gains, shared by all methods with the same seed."""
    rng = np.random.default_rng(cfg["seed"] + 7777)
    offs = rng.permutation(np.linspace(-cfg["spread"] / 2, cfg["spread"] / 2, cfg["K"]))
    return 10 ** ((cfg["snr"] + offs) / 10), rng


def transmit(delta, snr_lin, g, gen):
    """Uncoded analog transmission of one update over a block-Rayleigh subchannel with zero-forcing equalisation:
    the per-coordinate noise after equalisation has standard deviation ||delta|| / sqrt(D * snr * g)."""
    D = sum(v.numel() for v in delta.values())
    nrm = torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values()))
    std = nrm / math.sqrt(D * snr_lin * g)
    return {k: v + torch.randn(v.shape, device=DEV, generator=gen) * std for k, v in delta.items()}


def evaluate(model, X, y, C):
    model.eval(); P = []
    with torch.no_grad():
        for s in range(0, len(y), 8192): P.append(torch.softmax(model(X[s:s + 8192])[0], -1))
    P = torch.cat(P); loss = Fn.nll_loss(torch.log(P + 1e-12), y).item()
    return loss, (P.argmax(1) == y).float().mean().item(), P.cpu().numpy()


def epsilon(cfg, sizes, T):
    """(epsilon, delta) per client: subsampled-Gaussian RDP for DP-SGD + Gaussian mechanism for prototypes."""
    if cfg["sigma"] <= 0: return float("inf")
    eps = []
    for nk in sizes:
        q = min(1.0, cfg["B"] / nk); steps = T * cfg["E"] * math.ceil(nk / cfg["B"])
        orders = ORDERS if q < 0.5 else [o for o in ORDERS if float(o).is_integer()]   # fractional orders need q < 0.5
        rdp = compute_rdp(q=q, noise_multiplier=cfg["sigma"], steps=steps, orders=orders)
        if cfg["sigma_p"] > 0 and cfg["lam"] > 0:
            rdp = rdp + np.array(orders) * T * 2 / (2 * cfg["sigma_p"] ** 2)   # sensitivity sqrt(2): sum and count
        eps.append(get_privacy_spent(orders=orders, rdp=rdp, delta=cfg["delta"])[0])
    return float(max(eps))


DEFAULT = dict(K=10, alpha=0.5, T=40, E=1, B=1024, lr=3e-3, sigma=2.0, clip=1.0, sigma_p=20.0, delta=1e-5,
               focal=0, gamma=2.0, cbbeta=0.0, lam=0.0, mu=0.0, agg="trust", kappa=1.0, nb=2.0, beta=0.5, eta_s=1.0,
               attention=1, multiscale=0, mal=0.0, boost=10.0, mode="fed", h=128, ema=0.0, channel="ideal", snr=10.0, spread=10.0, gth=0.05, chw=0, seed=0, protocol="merged", save_probs=0)


def main():
    ds, tag = sys.argv[1], sys.argv[2]; cfg = dict(DEFAULT)
    for a in sys.argv[3:]:
        k, v = a.split("="); cfg[k] = type(DEFAULT[k])(v) if not isinstance(DEFAULT[k], str) else v
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    D = load(ds, cfg["protocol"]); C = len(D["classes"])
    Xtr, ytr = torch.tensor(D["Xtr"], device=DEV), torch.tensor(D["ytr"], device=DEV)
    Xva, yva = torch.tensor(D["Xva"], device=DEV), torch.tensor(D["yva"], device=DEV)
    Xte, yte = torch.tensor(D["Xte"], device=DEV), torch.tensor(D["yte"], device=DEV)
    model = PrismNet(Xtr.shape[1], C, h=cfg["h"], attention=bool(cfg["attention"]), multiscale=bool(cfg["multiscale"])).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    parts = dirichlet_partition(D["ytr"], cfg["K"], cfg["alpha"], seed=cfg["seed"])
    sizes = [len(p) for p in parts]
    nmal = int(round(cfg["mal"] * cfg["K"])); mal = set(range(cfg["K"] - nmal, cfg["K"]))
    hist = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], trust=[], time=[])
    t0 = time.time()

    if cfg["mode"] in ("central", "local"):
        groups = [np.arange(len(ytr))] if cfg["mode"] == "central" else parts
        epochs = cfg["T"] * cfg["E"] // (4 if cfg["mode"] == "central" else 1)
        accs, last_P = [], None
        for gi, g in enumerate(groups):
            m = copy.deepcopy(model); st = copy.deepcopy(m.state_dict()); c2 = dict(cfg, E=1, mu=0, lam=0)
            Xg, yg = Xtr[torch.tensor(g, device=DEV)], ytr[torch.tensor(g, device=DEV)]
            for ep in range(epochs):
                d, _, _, tl, ta = local_train(m, st, Xg, yg, c2, None, C=C)
                st = {k: st[k] + d[k] for k in st}; m.load_state_dict(st)
                if gi == 0 and ep % max(1, epochs // cfg["T"]) == 0 and len(hist["val_acc"]) < cfg["T"]:
                    vl, va, _ = evaluate(m, Xva, yva, C)
                    for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)): hist[k].append(v)
            _, a, P = evaluate(m, Xte, yte, C); accs.append(a); last_P = P if last_P is None else last_P
            if cfg["mode"] == "local":   # per-client model on the global test set; keep the probabilities of the mean client
                pass
        P = last_P; gstate = st
    else:
        gstate = copy.deepcopy(model.state_dict()); protos = None; mom = None; ema = None
        snr_k, rng_ch = channel_setup(cfg)
        gen_ch = torch.Generator(device=DEV); gen_ch.manual_seed(cfg["seed"] + 991)
        hist["received"] = []; hist["snr_db"] = (10 * np.log10(snr_k)).tolist()
        for r in range(cfg["T"]):
            gains = rng_ch.exponential(1.0, cfg["K"])   # |h_k|^2 for this round, drawn before training
            deltas, ps, pc, tl, ta = [], [], [], [], []
            for k, idx in enumerate(parts):
                ii = torch.tensor(idx, device=DEV)
                d, s, c, l, a = local_train(model, gstate, Xtr[ii], ytr[ii], cfg, protos, malicious=(k in mal), C=C)
                deltas.append(d); ps.append(s); pc.append(c); tl.append(l); ta.append(a)
            if cfg["channel"] == "rayleigh":   # truncated transmission: sites in a deep fade are in outage this round
                rx = [k for k in range(cfg["K"]) if gains[k] >= cfg["gth"]]
                deltas = [transmit(deltas[k], snr_k[k], gains[k], gen_ch) for k in rx]
                rho = [snr_k[k] * gains[k] / (1 + snr_k[k] * gains[k]) for k in rx]
            else:
                rx = list(range(cfg["K"])); rho = None
            hist["received"].append(len(rx))
            if not rx:   # every site in outage: the global model is kept for this round
                agg, trust = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gstate.items()}, []
            else:
                agg, trust = aggregate(deltas, [sizes[k] for k in rx], cfg, rho)
            ps = [ps[k] for k in rx]; pc = [pc[k] for k in rx]
            # integrity-weighted prototype pooling: a site whose update is rejected cannot steer the prototypes either
            psum = sum(t * s for t, s in zip(trust, ps)); pcnt = sum(t * c for t, c in zip(trust, pc))
            if cfg["beta"] > 0:   # server momentum
                mom = agg if mom is None else {k: cfg["beta"] * mom[k] + agg[k] for k in agg}
                step = mom
            else:
                step = agg
            gstate = {k: (gstate[k].float() + cfg["eta_s"] * step[k]).to(gstate[k].dtype) for k in gstate}
            if cfg["lam"] > 0:
                protos = Fn.normalize(psum / torch.clamp(pcnt, min=1.0)[:, None], dim=-1) * (pcnt[:, None] > 1).float()
            if cfg["ema"] > 0:   # server-side exponential moving average of the global weights (released model)
                ema = {k: v.clone() for k, v in gstate.items()} if ema is None else                     {k: (cfg["ema"] * ema[k].float() + (1 - cfg["ema"]) * gstate[k].float()).to(gstate[k].dtype) for k in gstate}
                model.load_state_dict(ema)
            else:
                model.load_state_dict(gstate)
            vl, va, _ = evaluate(model, Xva, yva, C)
            w = np.array(sizes) / sum(sizes)
            honest = [i for i in range(cfg["K"]) if i not in mal]
            hist["train_loss"].append(float(np.average(np.array(tl)[honest], weights=w[honest])))
            hist["train_acc"].append(float(np.average(np.array(ta)[honest], weights=w[honest])))
            hist["val_loss"].append(vl); hist["val_acc"].append(va); hist["trust"].append(trust); hist["time"].append(time.time() - t0)
            print(f"[{ds}/{tag}] r{r + 1:02d} train {hist['train_loss'][-1]:.4f}/{hist['train_acc'][-1]:.4f} val {vl:.4f}/{va:.4f} {time.time() - t0:.0f}s", flush=True)
        if ema is not None: gstate = ema
        model.load_state_dict(gstate); _, _, P = evaluate(model, Xte, yte, C); accs = None

    yt = D["yte"]; yp = P.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f, pc_n = precision_recall_fscore_support(yt, yp, average=None, labels=list(range(C)), zero_division=0)
    yb, pb = (yt > 0).astype(int), 1 - P[:, 0]
    tn, fp, fn, tp = confusion_matrix(yb, (yb * 0 + (yp > 0)).astype(int), labels=[0, 1]).ravel()
    try: auc = roc_auc_score(yt, P, multi_class="ovr", average="macro")
    except ValueError: auc = float("nan")
    res = dict(dataset=ds, tag=tag, cfg=cfg, params=nparams, sizes=sizes, malicious=sorted(mal),
               client_hist=[np.bincount(D["ytr"][p], minlength=C).tolist() for p in parts],
               test=dict(acc=accuracy_score(yt, yp), prec=pr, rec=rc, f1=f1, f1w=f1w, auc=auc,
                         bin_acc=float((tp + tn) / (tp + tn + fp + fn)), dr=float(tp / (tp + fn)), far=float(fp / (fp + tn)),
                         bin_auc=float(roc_auc_score(yb, pb)),
                         per_class=dict(prec=pc_p.tolist(), rec=pc_r.tolist(), f1=pc_f.tolist(), n=pc_n.tolist()),
                         cm=confusion_matrix(yt, yp, labels=list(range(C))).tolist(),
                         local_accs=accs),
               epsilon=epsilon(cfg, sizes, cfg["T"]) if cfg["mode"] == "fed" else None,
               hist=hist, seconds=time.time() - t0, classes=D["classes"].tolist())
    json.dump(res, open(RUNS / f"{ds}_{tag}.json", "w"), indent=1)
    if cfg["save_probs"]:
        np.save(RUNS / f"{ds}_{tag}_probs.npy", P.astype(np.float16)); torch.save(gstate, RUNS / f"{ds}_{tag}.pt")
    print(f"DONE {ds}/{tag} acc={res['test']['acc']:.4f} f1={f1:.4f} bin_acc={res['test']['bin_acc']:.4f} far={res['test']['far']:.4f} eps={res['epsilon']} t={res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
