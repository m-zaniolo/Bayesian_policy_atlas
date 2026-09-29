# Bayesian Policy Atlas

Reservoir operating policies that adapt to a **changing climate** by treating the
climate–demand regime as a *hidden state to be inferred*, rather than a value to
be plugged in.

The method has four stages, and the repository is laid out to match them:

| Stage | What it does | Code |
|---|---|---|
| **1. Climate–demand exposure** | Define a 2-D exposure space: flood magnitude (κ) × demand multiplier | `pipeline/s1_exposure.py` |
| **2. Policy atlas** | Optimise one policy per exposure cell → a continuous atlas of policies | `pipeline/s2_atlas.py` |
| **3. Bayesian state estimation** | Particle filter infers where in the exposure space we currently are | `pipeline/s3_belief.py` |
| **4. Bayesian action selection** | Turn the posterior into a release decision | `pipeline/s4_action.py` |

Stage 4 compares three ways of using the posterior:

- **mean-exposure (plug-in)** — use the atlas policy at the posterior *mean*; discards spread
- **posterior-optimal** — `argmin_i E_w[J_i]`, the policy minimising expected cost over the posterior
- **posterior-averaged** — apply the posterior-expected *decision*: release `Σ_j w_j · release_j`, a
  weighted average of **releases**, not of policy parameters

## Install

```bash
pip install -r requirements.txt
bash data/fetch_data.sh          # inputs are NOT redistributed; see UPSTREAM.md
```

There is no package to install. Scripts put `bdps/` on `sys.path` themselves, so
run them from the repository root:

```bash
python pipeline/s2_atlas.py --help
```

Configure paths with environment variables (see `bdps/paths.py`):

```bash
export BPA_DATA_ROOT=/scratch/$USER/bpa_data   # inputs  (default: ./data)
export BPA_RESULTS=/scratch/$USER/bpa_results  # outputs (default: ./results)
```

## Reproducing the results

Experiments span **9 reservoirs × 240 trajectories** (30 GCM × 8 demand
scenarios). The decision horizon is **h = 1 year** throughout.

Costs are `J = J_flood + J_shortage`, where shortage is squared deficit and
flood is a linear penalty on release above the safe threshold.

| Step | Command | Cost |
|---|---|---|
| Policy atlas | `python pipeline/s2_atlas.py` | hours–days (HPC; cached) |
| Belief calibration | `python pipeline/s3_belief.py` | minutes |
| Deployment (3 rules) | `python pipeline/s4_action.py` | ~1 h on 24 cores |
| Baselines + atlas oracle | `python benchmarks/run_baselines.py` | ~64 core-hours |
| Free-parameter oracle | `python benchmarks/oracle_freeparam.py --jobs 24` | ~12 core-hours |
| Perfect-foresight oracle | `python benchmarks/oracle_perfect_batch.py --jobs 24` | ~110 core-hours |
| Per-year cost streams | `python benchmarks/cost_over_time.py` | ~1 h on 24 cores |
| All figures | `python figures/fig*.py` | seconds |

Every long step writes a checkpoint and supports `--resume`, so runs can be
interrupted. Small summary CSVs are committed under `results/`, so the
figures reproduce without re-running the expensive stages.

Licensed MIT (see [LICENSE](LICENSE)).
