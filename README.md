# Byzantine Attacks and Robust Aggregation Defenses in Federated Learning

Course project – Cryptography and Network Security.
Pure **NumPy** implementation (no GPU / PyTorch needed). Everything is reproducible with seeds.

## 1. Project structure
| File | Purpose |
|---|---|
| `data.py` | Loads *digits* (offline) or *MNIST* (via OpenML), IID / Dirichlet non-IID partitioning |
| `model.py` | MLP with manual backprop; weights stored as a flat vector |
| `attacks.py` | 8 attack settings: none, label_flip, sign_flip, gaussian, scaling, ipm, alie, minmax |
| `defenses.py` | 9 aggregators: FedAvg, coordinate-wise Median, Trimmed Mean, Krum, Multi-Krum, Bulyan, Geometric Median (RFA), Norm-Clipping, FLTrust |
| `fl.py` | FL simulator (local SGD, malicious clients, aggregation, metrics) |
| `secure_agg.py` | **Crypto part**: pairwise-masking secure aggregation + HMAC-authenticated updates |
| `run_experiments.py` | Runs the attack × defense × Byzantine-fraction × seed grid → `results/*.csv` |
| `plot_results.py` | Heatmap, accuracy-vs-fraction, convergence curves, detection tables |

## 2. Setup & run
```bash
pip install -r requirements.txt
python secure_agg.py                          # crypto demo (masks cancel, HMAC verify)
python run_experiments.py --quick --out results_quick   # ~30 s sanity check
python run_experiments.py --alpha 0.5 --out results_noniid   # full grid, non-IID (~10 min)
python run_experiments.py --out results_iid                  # full grid, IID
python plot_results.py --dir results_noniid --byz 0.3
# MNIST (internet needed once):  python run_experiments.py --dataset mnist --alpha 0.5 --rounds 40
```
Useful flags: `--attacks`, `--defenses`, `--byz 0.1 0.2 0.3`, `--seeds`, `--clients`, `--rounds`, `--local-epochs`, `--lr`.

## 3. Threat model
* `n = 20` clients, a fraction `f/n` ∈ {10 %, 20 %, 30 %} is Byzantine (chosen randomly).
* Each round all clients train locally (2 epochs SGD) and upload an update `Δ = w_local − w_global`.
* Defenses are told the bound `f` (as Krum/Bulyan/Trimmed-Mean require).
* **Data poisoning**: label_flip (y → 9−y). **Model poisoning**: sign_flip, gaussian, scaling.
* **Omniscient / adaptive** (see all benign updates): IPM, ALIE, Min-Max.

## 4. Attacks implemented
| Attack | Idea | Reference |
|---|---|---|
| label_flip | train honestly on flipped labels | Tolpegin et al. 2020 |
| sign_flip | send −3·(honest update) | Damaskinos et al. |
| gaussian | random N(0,1) vector | Blanchard et al. 2017 |
| scaling | −10·mean(benign): huge magnitude | Bagdasaryan et al. 2020 (boosting idea) |
| ipm | −0.5·mean(benign) (small ε targets median-type rules; ε≫1 behaves like `scaling`) | Xie et al. 2020 |
| alie | μ − zσ staying inside benign variance | Baruch et al. 2019 |
| minmax | max-γ perturbation hidden inside benign cluster | Shejwalkar & Houmansadr, NDSS 2021 |

## 5. Defenses implemented
FedAvg (baseline, no defense) · coordinate-wise **Median** · **Trimmed Mean** (Yin et al. 2018) · **Krum / Multi-Krum** (Blanchard 2017) · **Bulyan** (El Mhamdi 2018) · **Geometric Median / RFA** (Pillutla 2022) · **Norm clipping** · **FLTrust** (Cao 2021, uses a 100-sample clean root dataset on the server).

## 6. Cryptography link (what to emphasise in the report)
* `secure_agg.py` implements Bonawitz-style **pairwise additive masking** over Z_2^32 with an HMAC-SHA256 PRG: the server sees only random-looking vectors, yet masks cancel in the sum.
* **Key insight – privacy vs robustness tension**: Krum/Median/Bulyan need to see *individual* updates, but secure aggregation hides them. A Byzantine client can also poison the sum undetectably (demo prints the error). Real systems need robust secure aggregation (e.g. ByzSecAgg, RoFL with zero-knowledge range proofs).
* **HMAC authentication** (`sign_update/verify_update`) stops tampering/impersonation in transit, but an *authenticated* insider can still be Byzantine → authentication ≠ robustness.
* Simplification: pairwise secrets come from `secrets` instead of ECDH; no dropout recovery (no Shamir sharing).

## 7. Metrics
Final & best test accuracy (mean over seeds), convergence curves, and the **fraction of selected updates that were Byzantine** for selection-based rules (Krum, Multi-Krum, Bulyan, FLTrust).

## 8. Expected findings (verify against your own results)
* FedAvg collapses under sign_flip / scaling / ipm_large / gaussian (accuracy → ~10–50 %).
* Median/Trimmed-Mean/Krum defeat crude attacks but degrade under **ALIE, IPM(ε=0.5), Min-Max**, especially with non-IID data.
* Krum performs poorly even without attacks on non-IID data (picks one client's update) – Multi-Krum is much better.
* FLTrust is the most robust here because the server has trusted root data; the price is that assumption.
* Higher Byzantine fraction and higher heterogeneity (smaller Dirichlet α) make all defenses worse.
* Robust rules cost accuracy when there is no attack (small price for robustness).

## 9. Suggested report outline
1. Introduction – FL, security threats, Byzantine failures  2. Background – FedAvg, threat model
3. Attacks  4. Defenses  5. Secure aggregation & authentication (crypto link)
6. Experimental setup  7. Results (heatmap, curves, fraction plots)  8. Discussion (privacy-robustness trade-off, limits)
9. Conclusion & future work (adaptive attacks on FLTrust, robust secure aggregation, differential privacy)
