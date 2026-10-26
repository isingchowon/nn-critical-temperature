"""
Multi-seed CNN experiment for the 3D Ising model (literature T_c = 4.5115).
Paste this whole cell into Google Colab (GPU runtime) and run it. Nothing else is needed.

Procedure (thesis, Sec. V):
  * Monte Carlo in C: simple cubic lattice, periodic boundaries, each step = 1 Metropolis sweep + 5 Wolff
    clusters; every (L, T) starts from the ordered state, 1000 equilibration steps, then one stored configuration
    every 10 steps until 1000 configurations are collected.
  * Three networks, each trained for every window (A, B, C) and every seed in SEEDS:
      - cnn_per_size : standard CNN, trained separately for each L
      - cnn_shared   : the same CNN, trained once on all sizes (batches of different L interleaved)
      - cnn_rescaled : scale-invariant CNN, sees only |M| L^(beta/nu) with M the lattice mean of a flip-odd field
  * Read-outs: level crossing P = 1/2 and its extrapolation T*(L) = T_inf + a L^(-1/nu) (nu fixed), size crossings
    of successive sizes; baselines: Binder cumulant and <|m|> L^(beta/nu). Bootstrap over held-out configurations.
  * Exponent test: the rescaled CNN and the <|m|> L^(beta/nu) baseline repeated with beta/nu = 0.1 (one seed).
Results are saved after every seed (RESULTS_FILE) and the run resumes if the cell is executed again.
"""
import os, json, time, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- configuration -----------------------------
SEEDS   = [0, 1, 2, 3, 4]
SIZES   = [8, 12, 16, 20]
TEMPS   = np.round(np.concatenate([np.arange(3.6, 4.21, 0.1),        # 7 values, 0.1 apart
                                   np.arange(4.30, 4.701, 0.02),     # 21 values, 0.02 apart around T_c
                                   np.arange(4.8, 5.51, 0.1)]), 4)   # 8 values  -> 36 temperatures
N_EQ, N_CONF, EVERY, N_WOLFF = 1000, 1000, 10, 5
EPOCHS, BATCH, LR, WEIGHT_DECAY = 10, 64, 1e-3, 1e-4
TEST_FRACTION, N_BOOT, SPLIT_SEED = 0.3, 300, 0
TC_REF, NU, BETA_NU, BETA_NU_WRONG = 4.5115, 0.630, 0.518, 0.1
LEVEL_RANGE = (4.0, 5.2)      # search interval for P = 1/2
CROSS_RANGE = (4.0, 5.0)      # size crossings of network outputs
BASE_RANGE  = (4.3, 4.7)      # size crossings of the baselines
EXPERIMENTS = ["cnn_per_size", "cnn_shared", "cnn_rescaled"]

def _window(lo, hi):
    return [int(i) for i in np.where((TEMPS >= lo - 1e-9) & (TEMPS <= hi + 1e-9))[0]]
WINDOWS = {"A": (_window(3.6, 3.9), _window(5.2, 5.5)),
           "B": (_window(3.6, 3.9), _window(4.8, 5.1)),
           "C": (_window(3.9, 4.2), _window(5.2, 5.5))}

USE_DRIVE = True
DATA_DIR, RESULTS_FILE, FIGURE_FILE = "ising_data", "ising_multiseed_results.json", "ising_multiseed.png"
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = "/content/drive/MyDrive/ising_thesis"
        os.makedirs(BASE, exist_ok=True)
        DATA_DIR, RESULTS_FILE, FIGURE_FILE = f"{BASE}/ising_data", f"{BASE}/{RESULTS_FILE}", f"{BASE}/{FIGURE_FILE}"
    except ImportError:
        print("not running in Colab: data and results are kept in the current directory")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_SOURCE = r'''// 3D Ising model, J = 1, periodic boundaries. Each step: one Metropolis sweep + n_wolff Wolff clusters.
// Usage: ./ising_gen L n_temps temps_file n_eq n_conf every n_wolff prefix seed
// Output: prefix_spins.bin int8 [n_temps][n_conf][L^3],  prefix_m.bin float32 [n_temps][n_conf] (magnetization per spin)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
static uint64_t st;
static inline uint64_t rng_next(void){ uint64_t x=st; x^=x>>12; x^=x<<25; x^=x>>27; st=x; return x*2685821657736338717ULL; }
static inline double rng_u(void){ return (rng_next()>>11)*(1.0/9007199254740992.0); }
int main(int argc,char**argv){
  if(argc<10){ fprintf(stderr,"usage: L n_temps temps_file n_eq n_conf every n_wolff prefix seed\n"); return 1; }
  int L=atoi(argv[1]),K=atoi(argv[2]); const char*tf=argv[3];
  int neq=atoi(argv[4]),nconf=atoi(argv[5]),every=atoi(argv[6]),nw=atoi(argv[7]);
  const char*prefix=argv[8]; st=strtoull(argv[9],0,10)*0x9E3779B97F4A7C15ULL+1; for(int w=0;w<20;w++) rng_next();
  int N=L*L*L; double*T=malloc(K*sizeof(double)); FILE*f=fopen(tf,"r");
  for(int k=0;k<K;k++) if(fscanf(f,"%lf",&T[k])!=1) return 2; fclose(f);
  int*nb=malloc((size_t)6*N*sizeof(int));
  for(int x=0;x<L;x++) for(int y=0;y<L;y++) for(int z=0;z<L;z++){
    int i=(x*L+y)*L+z;
    nb[6*i+0]=(((x+1)%L)*L+y)*L+z;   nb[6*i+1]=(((x+L-1)%L)*L+y)*L+z;
    nb[6*i+2]=(x*L+(y+1)%L)*L+z;     nb[6*i+3]=(x*L+(y+L-1)%L)*L+z;
    nb[6*i+4]=(x*L+y)*L+(z+1)%L;     nb[6*i+5]=(x*L+y)*L+(z+L-1)%L; }
  int8_t*s=malloc(N),*buf=malloc((size_t)nconf*N); int*stack=malloc(N*sizeof(int)); char*in=malloc(N);
  float*mag=malloc((size_t)K*nconf*sizeof(float));
  char fn[512]; snprintf(fn,512,"%s_spins.bin",prefix); FILE*fs=fopen(fn,"wb");
  for(int k=0;k<K;k++){
    double b=1.0/T[k], padd=1.0-exp(-2.0*b), acc[13];
    for(int d=-6;d<=6;d++) acc[d+6]=exp(-2.0*b*d);       // Metropolis: dE = 2 s_i h_i, h_i in {-6,...,6}
    for(int i=0;i<N;i++) s[i]=1;                          // ordered start
    int total=neq+nconf*every, c=0;
    for(int t=0;t<total;t++){
      for(int i=0;i<N;i++){ int h=0; for(int a=0;a<6;a++) h+=s[nb[6*i+a]];
        int d=s[i]*h; if(d<=0 || rng_u()<acc[d+6]) s[i]=-s[i]; }
      for(int w=0;w<nw;w++){                              // Wolff cluster
        int seed=(int)(rng_u()*N); if(seed>=N) seed=N-1; int8_t sv=s[seed]; int top=0;
        stack[top++]=seed; s[seed]=-sv;
        while(top){ int i=stack[--top];
          for(int a=0;a<6;a++){ int j=nb[6*i+a]; if(s[j]==sv && rng_u()<padd){ s[j]=-sv; stack[top++]=j; } } }
      }
      if(t>=neq && (t-neq)%every==every-1 && c<nconf){
        long m=0; for(int i=0;i<N;i++){ m+=s[i]; buf[(size_t)c*N+i]=s[i]; }
        mag[(size_t)k*nconf+c]=(float)m/N; c++; }
    }
    fwrite(buf,1,(size_t)nconf*N,fs);
    fprintf(stderr,"L=%d T=%.3f done\n",L,T[k]);
  }
  fclose(fs); (void)in;
  snprintf(fn,512,"%s_m.bin",prefix); f=fopen(fn,"wb"); fwrite(mag,sizeof(float),(size_t)K*nconf,f); fclose(f);
  return 0;
}
'''


# ----------------------------- data -----------------------------
def generate_all():
    """Compile the generator and run it for every size without data (sizes in parallel)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists("./ising_gen"):
        with open("ising_gen.c", "w") as f:
            f.write(C_SOURCE)
        subprocess.run(["gcc", "-O3", "-march=native", "-o", "ising_gen", "ising_gen.c", "-lm"], check=True)
    np.savetxt(f"{DATA_DIR}/temps.txt", TEMPS)
    procs = []
    for i, L in enumerate(SIZES):
        if os.path.exists(f"{DATA_DIR}/L{L}.done"):
            continue
        cmd = ["./ising_gen", str(L), str(len(TEMPS)), f"{DATA_DIR}/temps.txt", str(N_EQ), str(N_CONF),
               str(EVERY), str(N_WOLFF), f"{DATA_DIR}/L{L}", str(200 + i)]
        print("running:", " ".join(cmd), flush=True)
        procs.append((L, subprocess.Popen(cmd)))
    for L, p in procs:
        if p.wait() != 0:
            raise RuntimeError("ising_gen failed")
        open(f"{DATA_DIR}/L{L}.done", "w").close()


def load(L):
    K = len(TEMPS)
    spins = np.fromfile(f"{DATA_DIR}/L{L}_spins.bin", dtype=np.int8).reshape(K, N_CONF, L, L, L)
    m = np.fromfile(f"{DATA_DIR}/L{L}_m.bin", dtype=np.float32).reshape(K, N_CONF).astype(np.float64)
    perm = np.random.default_rng(SPLIT_SEED + L).permutation(N_CONF)         # fixed split, same for every T
    n_test = int(round(TEST_FRACTION * N_CONF))
    return dict(L=L, spins=torch.from_numpy(spins).to(device), m=m,
                test=np.sort(perm[:n_test]), train=np.sort(perm[n_test:]))


# ----------------------------- networks -----------------------------
class StandardCNN(nn.Module):
    """Two 3x3x3 convolutions (32 channels, periodic padding, ReLU), global average pooling, MLP 32-32-1."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv3d(1, 32, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv3d(32, 32, 3, padding=1, padding_mode="circular")
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x))))
        return self.head(h.mean(dim=(2, 3, 4))).squeeze(-1)


class ScaleInvariantCNN(nn.Module):
    """phi_x = sum_c w_c tanh(sum_u W_cu s_{x+u}) (no biases, so phi is odd under s -> -s);
    M = lattice mean of phi; only |M| L^(beta/nu) reaches the MLP 1-16-1."""
    def __init__(self, beta_nu):
        super().__init__()
        self.beta_nu = beta_nu
        self.conv = nn.Conv3d(1, 32, 3, padding=1, padding_mode="circular", bias=False)
        self.mix = nn.Conv3d(32, 1, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        L = x.shape[-1]
        M = self.mix(torch.tanh(self.conv(x))).mean(dim=(1, 2, 3, 4))
        return self.head((M.abs() * L ** self.beta_nu).unsqueeze(-1)).squeeze(-1)


def make_model(name, beta_nu):
    return ScaleInvariantCNN(beta_nu).to(device) if name == "cnn_rescaled" else StandardCNN().to(device)


def train(model, data, sizes, window, seed):
    low, high = window
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    ex = {}
    for L in sizes:
        k, c = np.meshgrid(low + high, data[L]["train"], indexing="ij")
        y = np.isin(k, low).astype(np.float32)                       # 1 = ordered, 0 = disordered
        ex[L] = [torch.from_numpy(a.ravel()).to(device) for a in (k, c, y)]
    rng = np.random.default_rng(seed)                                # batch order depends on the seed
    model.train()
    for epoch in range(EPOCHS):
        batches = []
        for L in sizes:
            p = rng.permutation(len(ex[L][0]))
            batches += [(L, p[i:i + BATCH]) for i in range(0, len(p), BATCH)]
        for j in rng.permutation(len(batches)):                      # sizes interleaved at random
            L, idx = batches[j]
            idx = torch.from_numpy(idx).to(device)
            k, c, y = [a[idx] for a in ex[L]]
            x = data[L]["spins"][k, c].float().unsqueeze(1)
            flip = (torch.randint(0, 2, (len(idx), 1, 1, 1, 1), device=device) * 2 - 1).float()
            loss = loss_fn(model(x * flip), y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def predict(model, d):
    """P(ordered) for every held-out configuration and temperature: array (n_test, K)."""
    model.eval()
    test = torch.from_numpy(d["test"]).to(device)
    P = np.zeros((len(test), len(TEMPS)))
    for k in range(len(TEMPS)):
        for i in range(0, len(test), 256):
            x = d["spins"][k, test[i:i + 256]].float().unsqueeze(1)
            P[i:i + 256, k] = torch.sigmoid(model(x)).cpu().numpy()
    return P


def train_predict(name, window, seed, data, beta_nu=BETA_NU):
    """Returns {L: P (n_test, K)} for one experiment, window and seed."""
    out = {}
    groups = [[L] for L in SIZES] if name == "cnn_per_size" else [SIZES]
    for sizes in groups:
        torch.manual_seed(seed)                                      # initialisation depends on the seed
        model = make_model(name, beta_nu)
        train(model, data, sizes, window, seed)
        for L in sizes:
            out[L] = predict(model, data[L])
    return out


# ----------------------------- T_c estimators -----------------------------
def crossing(T, y, lo, hi, ref=None, target=0.5):
    """Temperature in [lo, hi] where y changes sign. Several: with `ref` the one where ref is closest to
    `target` (discards noise crossings in the flat tails); without `ref` the steepest one."""
    best, score = None, 1e9
    for k in range(len(T) - 1):
        if y[k] * y[k + 1] < 0:
            f = y[k] / (y[k] - y[k + 1])
            t = T[k] + (T[k + 1] - T[k]) * f
            sc = -abs((y[k + 1] - y[k]) / (T[k + 1] - T[k])) if ref is None else abs(ref[k] + f * (ref[k + 1] - ref[k]) - target)
            if lo <= t <= hi and sc < score:
                best, score = t, sc
    return best


def bootstrap(fn, sizes_n, rng):
    vals = [fn([rng.integers(0, n, n) for n in sizes_n]) for _ in range(N_BOOT)]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if len(vals) > N_BOOT / 2 else (float("nan"), float("nan"))


def extrapolate(tstar, err):
    """Weighted fit T*(L) = T_inf + a L^(-1/nu), nu fixed. Returns (T_inf, error)."""
    x = np.array(SIZES, float) ** (-1.0 / NU)
    w = 1.0 / np.maximum(np.array(err), 1e-3) ** 2
    A = np.vstack([np.ones(len(x)), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ A.T @ (w * np.array(tstar))
    return float(coef[0]), float(cov[0, 0] ** 0.5)


def analyse(P, window, rng):
    T = TEMPS
    low, high = window
    pairs = list(zip(SIZES[:-1], SIZES[1:]))
    out = {"acc": float(np.mean([np.mean(np.concatenate([P[L][:, low].ravel() > 0.5, P[L][:, high].ravel() < 0.5]))
                                 for L in SIZES]))}
    half = {L: bootstrap(lambda ix: crossing(T, P[L][ix[0]].mean(0) - 0.5, *LEVEL_RANGE), [len(P[L])], rng)
            for L in SIZES}
    out["half"] = {str(L): list(half[L]) for L in SIZES}
    out["T_inf"] = (list(extrapolate([half[L][0] for L in SIZES], [half[L][1] for L in SIZES]))
                    if all(np.isfinite(half[L][0]) for L in SIZES) else [float("nan")] * 2)

    def cross(a, b):
        def fn(ix):
            pa, pb = P[a][ix[0]].mean(0), P[b][ix[1]].mean(0)
            return crossing(T, pa - pb, *CROSS_RANGE, (pa + pb) / 2, 0.5)
        return bootstrap(fn, [len(P[a]), len(P[b])], rng)
    out["cross"] = {f"{a}-{b}": list(cross(a, b)) for a, b in pairs}
    out["P"] = {str(L): P[L].mean(0).tolist() for L in SIZES}
    return out


def baselines(data, beta_nu, rng):
    """Binder cumulant and <|m|> L^(beta/nu) crossings (held-out configurations, bootstrap)."""
    T = TEMPS
    m = {L: data[L]["m"][:, data[L]["test"]].T for L in SIZES}         # (n_test, K)
    def binder(L, ix):
        x = m[L][ix]
        return 1.0 - (x ** 4).mean(0) / (3.0 * (x ** 2).mean(0) ** 2)
    def mres(L, ix):
        return np.abs(m[L][ix]).mean(0) * L ** beta_nu
    out = {"binder": {}, "m_rescaled": {}}
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        n = [len(m[a]), len(m[b])]
        out["binder"][f"{a}-{b}"] = list(bootstrap(lambda ix: crossing(T, binder(a, ix[0]) - binder(b, ix[1]), *BASE_RANGE), n, rng))
        out["m_rescaled"][f"{a}-{b}"] = list(bootstrap(lambda ix: crossing(T, mres(a, ix[0]) - mres(b, ix[1]), *BASE_RANGE), n, rng))
    return out


# ----------------------------- summary -----------------------------
def agg(pairs_list):
    v = np.array([p[0] for p in pairs_list], float); e = np.array([p[1] for p in pairs_list], float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return dict(mean=float("nan"), seed_std=float("nan"), boot=float("nan"), n=0)
    return dict(mean=float(v[ok].mean()), seed_std=float(v[ok].std(ddof=1)) if ok.sum() > 1 else 0.0,
                boot=float(e[ok].mean()), n=int(ok.sum()))


def show(a, n_total):
    if a["n"] == 0:
        return "not found"
    err = int(round(1000 * np.hypot(a["seed_std"], a["boot"])))
    return f"{a['mean']:.3f}({err})  [runs {a['seed_std']:.3f}, boot {a['boot']:.3f}, {a['n']}/{n_total}]"


def summarize(results):
    seeds = [s for s in map(str, SEEDS) if s in results["runs"]]
    pairs = [f"{a}-{b}" for a, b in zip(SIZES[:-1], SIZES[1:])]
    print("\n" + "=" * 100 + f"\nSUMMARY over {len(seeds)} seeds; value(error) with error = runs (+) bootstrap in "
          f"quadrature, last digits. Literature T_c = {TC_REF}\n" + "=" * 100)
    for name, r in results["baselines"].items():
        print(f"baseline {name:12s} " + "   ".join(f"({p}) {r[p][0]:.3f}({int(round(1000 * r[p][1]))})" for p in pairs))
    summary = {}
    for name in EXPERIMENTS:
        print(f"\n--- {name} ---")
        summary[name] = {}
        for w in WINDOWS:
            runs = [results["runs"][s][name][w] for s in seeds]
            S = {"T_inf": agg([r["T_inf"] for r in runs]),
                 "half": {str(L): agg([r["half"][str(L)] for r in runs]) for L in SIZES},
                 "cross": {p: agg([r["cross"][p] for r in runs]) for p in pairs},
                 "acc": float(np.mean([r["acc"] for r in runs]))}
            summary[name][w] = S
            print(f"  window {w}  accuracy {S['acc']:.3f}")
            print("     level crossings T*(L): " + "   ".join(f"L={L}: {show(S['half'][str(L)], len(seeds))}" for L in SIZES))
            print(f"     extrapolated T_inf:    {show(S['T_inf'], len(seeds))}")
            print("     size crossings:        " + "   ".join(f"({p}): {show(S['cross'][p], len(seeds))}" for p in pairs))
    if "exponent_test" in results:
        print(f"\n--- exponent test, beta/nu = {BETA_NU_WRONG}, seed {SEEDS[0]} (bootstrap errors) ---")
        et = results["exponent_test"]
        print("  <|m|> L^0.1: " + "   ".join(f"({p}) {et['m_rescaled'][p][0]:.3f}({int(round(1000 * et['m_rescaled'][p][1]))})" for p in pairs))
        for w in WINDOWS:
            c = et["cnn"][w]["cross"][pairs[-1]]
            print(f"  rescaled CNN window {w}, ({pairs[-1]}): {c[0]:.3f}({int(round(1000 * c[1]))})")
    results["summary"] = summary


# ----------------------------- main -----------------------------
def main():
    generate_all()
    data = {L: load(L) for L in SIZES}
    config = dict(SIZES=SIZES, TEMPS=TEMPS.tolist(), N_EQ=N_EQ, N_CONF=N_CONF, EVERY=EVERY, N_WOLFF=N_WOLFF,
                  EPOCHS=EPOCHS, BATCH=BATCH, TEST_FRACTION=TEST_FRACTION, SPLIT_SEED=SPLIT_SEED, BETA_NU=BETA_NU,
                  WINDOWS={w: [a, b] for w, (a, b) in WINDOWS.items()})
    print(f"device: {device}    seeds: {SEEDS}    literature T_c = {TC_REF}")
    for w, (a, b) in WINDOWS.items():
        print(f"window {w}: ordered T in {TEMPS[a].tolist()}, disordered T in {TEMPS[b].tolist()}")
    results = {"config": config, "T": TEMPS.tolist(), "runs": {}}
    if os.path.exists(RESULTS_FILE):
        old = json.load(open(RESULTS_FILE))
        if old.get("config") == config:
            results = old
            print(f"resuming: seeds already finished: {sorted(results['runs'], key=int)}")

    results["baselines"] = baselines(data, BETA_NU, np.random.default_rng(12345))
    for seed in SEEDS:
        if str(seed) in results["runs"]:
            continue
        t0 = time.time(); rng = np.random.default_rng(1000 + seed); run = {}
        for name in EXPERIMENTS:
            run[name] = {}
            for w, window in WINDOWS.items():
                r = analyse(train_predict(name, window, seed, data), window, rng)
                run[name][w] = r
                print(f"seed {seed}  {name:13s} window {w}:  T_inf = {r['T_inf'][0]:.3f}   size crossings "
                      + " ".join(f"{v[0]:.3f}" for v in r["cross"].values()), flush=True)
        results["runs"][str(seed)] = run
        json.dump(results, open(RESULTS_FILE, "w"))
        print(f"seed {seed} finished in {(time.time() - t0) / 60:.1f} min and saved to {RESULTS_FILE}\n", flush=True)

    if "exponent_test" not in results:                                   # deliberately wrong exponent, one seed
        rng = np.random.default_rng(999)
        et = {"m_rescaled": baselines(data, BETA_NU_WRONG, rng)["m_rescaled"], "cnn": {}}
        for w, window in WINDOWS.items():
            et["cnn"][w] = analyse(train_predict("cnn_rescaled", window, SEEDS[0], data, BETA_NU_WRONG), window, rng)
        results["exponent_test"] = et
        json.dump(results, open(RESULTS_FILE, "w"))

    summarize(results)
    json.dump(results, open(RESULTS_FILE, "w"))

    # figure: mean over seeds (line) and min-max over seeds (band)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = [s for s in map(str, SEEDS) if s in results["runs"]]
    fig, axes = plt.subplots(len(EXPERIMENTS), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 3.2 * len(EXPERIMENTS)), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(SIZES)))
    for i, name in enumerate(EXPERIMENTS):
        for j, w in enumerate(WINDOWS):
            ax = axes[i, j]
            for col, L in zip(colors, SIZES):
                curves = np.array([results["runs"][s][name][w]["P"][str(L)] for s in seeds])
                ax.fill_between(TEMPS, curves.min(0), curves.max(0), color=col, alpha=0.25, lw=0)
                ax.plot(TEMPS, curves.mean(0), "-", color=col, lw=1.4, label=f"L={L}")
            ax.axvline(TC_REF, color="k", ls="--", lw=1); ax.axhline(0.5, color="gray", lw=0.6)
            ax.set_xlim(4.2, 4.8); ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{name}, window {w}", fontsize=9); ax.set_xlabel("T"); ax.set_ylabel("P(ordered)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=180)
    print(f"\nsaved {RESULTS_FILE} and {FIGURE_FILE}")


main()
