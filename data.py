# -*- coding: utf-8 -*-
"""Load NSL-KDD and UNSW-NB15, encode, scale, split and cache as tensors.

Protocol: the official train and test files are merged and re-split 70/10/20 (train/val/test) with
stratification; the official-split variant keeps KDDTrain+/KDDTest+ and the UNSW training/testing sets."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; CACHE = ROOT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)

KDD_COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
DOS = "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split()
PROBE = "satan ipsweep nmap portsweep mscan saint".split()
R2L = "guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named".split()
U2R = "buffer_overflow loadmodule rootkit perl sqlattack xterm ps".split()
KDD_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
UNSW_CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]


def _kdd_cat(l):
    if l == "normal": return 0
    if l in DOS: return 1
    if l in PROBE: return 2
    if l in R2L: return 3
    if l in U2R: return 4
    raise ValueError(l)


def raw(name):
    """Returns (train_df, test_df, cat_cols, label_col_multiclass_int) for the official files."""
    if name == "nslkdd":
        tr = pd.read_csv(DATA / "nslkdd" / "KDDTrain+.txt", names=KDD_COLS)
        te = pd.read_csv(DATA / "nslkdd" / "KDDTest+.txt", names=KDD_COLS)
        for d in (tr, te):
            d["y"] = d["label"].map(_kdd_cat); d.drop(columns=["label", "difficulty", "num_outbound_cmds"], inplace=True)
        return tr, te, ["protocol_type", "service", "flag"], KDD_CLASSES
    if name == "unsw":
        tr = pd.read_csv(DATA / "unsw" / "UNSW_NB15_training-set.csv"); te = pd.read_csv(DATA / "unsw" / "UNSW_NB15_testing-set.csv")
        for d in (tr, te):
            d.columns = [c.strip().lstrip("﻿") for c in d.columns]
            d["y"] = d["attack_cat"].str.strip().map({c: i for i, c in enumerate(UNSW_CLASSES)})
            d.drop(columns=["id", "attack_cat", "label"], inplace=True)
        return tr, te, ["proto", "service", "state"], UNSW_CLASSES
    raise ValueError(name)


def encode(tr, te, cats):
    """One-hot categorical fields (vocabulary of the training part), log1p + z-score numeric fields."""
    num = [c for c in tr.columns if c not in cats + ["y"]]
    Xtr_n = np.log1p(tr[num].clip(lower=0).astype(float).values); Xte_n = np.log1p(te[num].clip(lower=0).astype(float).values)
    mu, sd = Xtr_n.mean(0), Xtr_n.std(0) + 1e-6
    Xtr_n = (Xtr_n - mu) / sd; Xte_n = (Xte_n - mu) / sd
    oh_tr, oh_te, names = [], [], list(num)
    for c in cats:
        vocab = sorted(tr[c].astype(str).unique())
        idx = {v: i for i, v in enumerate(vocab)}
        a = np.zeros((len(tr), len(vocab)), np.float32); b = np.zeros((len(te), len(vocab)), np.float32)
        a[np.arange(len(tr)), tr[c].astype(str).map(idx).values] = 1
        m = te[c].astype(str).map(idx); ok = m.notna().values
        b[np.arange(len(te))[ok], m[ok].astype(int).values] = 1
        oh_tr.append(a); oh_te.append(b); names += [f"{c}={v}" for v in vocab]
    Xtr = np.hstack([Xtr_n.astype(np.float32)] + oh_tr); Xte = np.hstack([Xte_n.astype(np.float32)] + oh_te)
    return Xtr, Xte, names


def load(name, protocol="merged", seed=0):
    f = CACHE / f"{name}_{protocol}_{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tr, te, cats, classes = raw(name)
    if protocol == "merged":
        full = pd.concat([tr, te], ignore_index=True)
        a, test = train_test_split(full, test_size=0.20, stratify=full["y"], random_state=seed)
        train, val = train_test_split(a, test_size=0.125, stratify=a["y"], random_state=seed)
    else:
        train, val = train_test_split(tr, test_size=0.10, stratify=tr["y"], random_state=seed); test = te
    Xtr, Xte, names = encode(train, pd.concat([val, test]), cats)
    Xva, Xte = Xte[:len(val)], Xte[len(val):]
    out = dict(Xtr=Xtr, ytr=train["y"].values.astype(np.int64), Xva=Xva, yva=val["y"].values.astype(np.int64),
               Xte=Xte, yte=test["y"].values.astype(np.int64), classes=np.array(classes), features=np.array(names))
    np.savez_compressed(f, **out)
    return out


def dirichlet_partition(y, K, alpha, seed=0, min_size=50):
    """Label-skewed non-IID split: for every class draw client shares from Dir(alpha)."""
    rng = np.random.default_rng(seed); C = y.max() + 1
    while True:
        parts = [[] for _ in range(K)]
        for c in range(C):
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, s in enumerate(np.split(idx, cuts)): parts[k] += s.tolist()
        if min(len(p) for p in parts) >= min_size: return [np.array(sorted(p)) for p in parts]
        seed += 1000; rng = np.random.default_rng(seed)


if __name__ == "__main__":
    for n in ("nslkdd", "unsw"):
        for pr in ("merged", "official"):
            d = load(n, pr)
            print(n, pr, d["Xtr"].shape, d["Xva"].shape, d["Xte"].shape, np.bincount(d["ytr"]), np.bincount(d["yte"]))
