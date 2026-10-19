"""Run attack x defense x byzantine-fraction grid and save results.

Examples
  python run_experiments.py --quick                      # ~ couple of minutes sanity run
  python run_experiments.py                              # default grid on digits
  python run_experiments.py --dataset mnist --rounds 40  # needs internet once
  python run_experiments.py --alpha 0.5                  # non-IID (Dirichlet 0.5)
"""
import argparse, itertools, json, os, time
import pandas as pd
from data import load_dataset
from fl import run_fl
from attacks import ATTACKS
from defenses import DEFENSES

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
p.add_argument("--attacks", nargs="+", default=list(ATTACKS))
p.add_argument("--defenses", nargs="+", default=list(DEFENSES))
p.add_argument("--byz", nargs="+", type=float, default=[0.1, 0.2, 0.3])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--clients", type=int, default=20)
p.add_argument("--rounds", type=int, default=30)
p.add_argument("--local-epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha; omit for IID")
p.add_argument("--out", default="results")
p.add_argument("--quick", action="store_true")
a = p.parse_args()

if a.quick:
    a.byz, a.seeds, a.rounds = [0.2], [0], 15
os.makedirs(a.out, exist_ok=True)
Xtr, ytr, Xte, yte = load_dataset(a.dataset)
rows, histories = [], {}
combos = [("none", d, 0.0) for d in a.defenses] if "none" in a.attacks else []
combos += [(at, d, b) for at in a.attacks if at != "none" for d in a.defenses for b in a.byz]
t0 = time.time()
for i, (at, d, b) in enumerate(combos):
    for s in a.seeds:
        h = run_fl(Xtr, ytr, Xte, yte, attack=at, defense=d, n_clients=a.clients, byz_frac=b if at != "none" else (a.byz[0]),
                   rounds=a.rounds, local_epochs=a.local_epochs, lr=a.lr, alpha=a.alpha, seed=s)
        rows.append(dict(dataset=a.dataset, attack=at, defense=d, byz_frac=b, seed=s,
                         final_acc=h["acc"][-1], best_acc=max(h["acc"]),
                         byz_selected=(sum(h["byz_selected"]) / len(h["byz_selected"])) if h["byz_selected"] else None))
        histories[f"{at}|{d}|{b}|{s}"] = h["acc"]
    print(f"[{i+1}/{len(combos)}] {at:10s} {d:13s} byz={b:.1f}  "
          f"final_acc={pd.DataFrame(rows).query('attack==@at and defense==@d and byz_frac==@b').final_acc.mean():.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/results.csv", index=False)
    json.dump(histories, open(f"{a.out}/histories.json", "w"))
print("saved to", a.out)
