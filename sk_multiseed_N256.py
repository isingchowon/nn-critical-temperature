"""
Multi-seed GNN + LSTM experiment for the Sherrington-Kirkpatrick spin glass (exact T_c = 1).
Paste this whole cell into Google Colab (GPU runtime) and run it. Nothing else is needed.

Compared with the single-run SK script this version
  * prints equilibration checks for the Monte Carlo data;
  * repeats every experiment and training window for each seed in SEEDS (data and train/test split stay fixed);
  * adds a control: a classifier that sees only the spins of one replica (no couplings), which should score 50%;
  * saves after every seed (RESULTS_FILE) and resumes if the cell is run again;
  * prints one summary table:  mean over seeds +- spread over seeds  (mean bootstrap error, seeds found / total).

N = 256 version: identical to the thesis script except that N = 256 is added to SIZES (with its own number of
samples and equilibration sweeps), data and results go to Google Drive so a disconnect does not lose them, and
the results/figure get new file names so the N <= 128 results are not overwritten.
The seeds of the existing sizes are unchanged, so N = 32, 64, 128 regenerate exactly the same data as before.
"""
import os, json, time, subprocess
import numpy as np
import torch
import torch.nn as nn

# ----------------------------- configuration -----------------------------
SEEDS   = [0, 1, 2, 3, 4]                    # independent training runs; use [0, 1, 2] if time is short
SIZES   = [32, 64, 128, 256]                 # number of spins N  (256 appended at the end: keeps the seeds of 32-128)
SAMPLES = {32: 300, 64: 300, 128: 200, 256: 200}       # disorder samples per size
N_EQ    = {32: 5000, 64: 5000, 128: 6000, 256: 12000}  # equilibration sweeps
K, TMIN, TMAX = 24, 0.5, 2.0                 # temperatures (geometric spacing)
N_MEAS, EVERY = 3200, 20                     # -> 160 stored measurements per (sample, T)
SEQ_LEN = 16                                 # snapshots per LSTM sequence -> 10 sequences per (sample, T)
HIDDEN, GNN_LAYERS, LSTM_HIDDEN = 32, 2, 32
EPOCHS, BATCH, LR, WEIGHT_DECAY = 6, 64, 1e-3, 1e-4
TEST_FRACTION, N_BOOT, SPLIT_SEED = 0.3, 300, 0
TC_REF = 1.0
USE_DRIVE = True                             # keep data and results on Google Drive (survives a disconnect)
DATA_DIR = "sk_data"
RESULTS_FILE = "sk_multiseed_N256_results.json"
FIGURE_FILE = "sk_multiseed_N256.png"
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = "/content/drive/MyDrive/sk_thesis"
        os.makedirs(BASE, exist_ok=True)
        DATA_DIR = f"{BASE}/sk_data"
        RESULTS_FILE = f"{BASE}/{RESULTS_FILE}"
        FIGURE_FILE = f"{BASE}/{FIGURE_FILE}"
    except ImportError:
        print("not running in Colab: data and results are kept in the current directory")
# training windows as (low-T indices, high-T indices) into the temperature grid: 4 + 4 temperatures each
WINDOWS = {"A": (range(0, 4), range(20, 24)),      # T in [0.50,0.60] and [1.67,2.00]
           "B": (range(0, 4), range(15, 19)),      # T in [0.50,0.60] and [1.23,1.48]
           "C": (range(5, 9), range(20, 24))}      # T in [0.68,0.81] and [1.67,2.00]
EXPERIMENTS = {  # name: (mode, use_gnn, rescale)
    "overlap_gnn_lstm": ("overlap", True,  True),
    "overlap_lstm":     ("overlap", False, True),
    "single_gnn_lstm":  ("single",  True,  False),
}
LEVEL_RANGE = (0.55, 1.9)     # search interval for the P = 0.5 level crossing
CROSS_RANGE = (0.6, 1.6)      # search interval for size crossings of the network output
BASE_RANGE  = (0.7, 1.4)      # search interval for the conventional baselines
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_SOURCE = r'''// Parallel-tempering Monte Carlo for the Sherrington-Kirkpatrick model.
// H = -sum_{i<j} J_ij s_i s_j,  J_ij ~ N(0, 1/N).  Two replicas (a, b) per temperature.
// Usage: ./sk_gen N S K Tmin Tmax n_eq n_meas every prefix seed      (M = n_meas / every)
// Output: prefix_J.bin     float32 [S][N][N]        couplings
//         prefix_spins.bin int8    [S][K][M][2][N]  both replicas at every measurement
//         prefix_q.bin     float32 [S][K][M]        overlap between the two replicas
//         prefix_e.bin     float32 [S][K][M]        energy per spin of replica a
//         prefix_T.txt     temperature, swap acceptance rate
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t rng_next(uint64_t *s){
  uint64_t x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x; return x*2685821657736338717ULL; }
static inline double rng_u(uint64_t *s){ return (rng_next(s)>>11)*(1.0/9007199254740992.0); }
static double rng_gauss(uint64_t *s){
  double u1=rng_u(s),u2=rng_u(s); if(u1<1e-300)u1=1e-300;
  return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2); }

static void fields(int N,const float*J,const float*s,float*h){
  for(int i=0;i<N;i++){ float a=0; const float*Ji=J+(size_t)i*N; for(int j=0;j<N;j++) a+=Ji[j]*s[j]; h[i]=a; } }
static double energy(int N,const float*s,const float*h){
  double e=0; for(int i=0;i<N;i++) e+=s[i]*h[i]; return -0.5*e; }

int main(int argc,char**argv){
  if(argc<11){ fprintf(stderr,"usage: N S K Tmin Tmax n_eq n_meas every prefix seed\n"); return 1; }
  int N=atoi(argv[1]),S=atoi(argv[2]),K=atoi(argv[3]); double Tmin=atof(argv[4]),Tmax=atof(argv[5]);
  int neq=atoi(argv[6]),nmeas=atoi(argv[7]),every=atoi(argv[8]);
  const char*prefix=argv[9]; uint64_t seed=strtoull(argv[10],0,10);
  int M=nmeas/every;
  double*T=malloc(K*sizeof(double)),*beta=malloc(K*sizeof(double));
  for(int k=0;k<K;k++){ T[k]=Tmin*pow(Tmax/Tmin,(double)k/(K-1)); beta[k]=1.0/T[k]; }
  float*q=malloc((size_t)S*K*M*sizeof(float)),*ea=malloc((size_t)S*K*M*sizeof(float));
  int8_t*buf=malloc((size_t)K*M*2*N);
  char fn[512]; snprintf(fn,512,"%s_J.bin",prefix); FILE*fJ=fopen(fn,"wb");
  snprintf(fn,512,"%s_spins.bin",prefix); FILE*fS=fopen(fn,"wb");
  float*J=malloc((size_t)N*N*sizeof(float));
  float*s=malloc((size_t)2*K*N*sizeof(float)),*h=malloc((size_t)2*K*N*sizeof(float));
  double*E=malloc(2*K*sizeof(double)); int*idx=malloc(2*K*sizeof(int));
  double*swap_acc=calloc(K,sizeof(double));
  for(int smp=0;smp<S;smp++){
    uint64_t st=seed*0x9E3779B97F4A7C15ULL+(uint64_t)(smp+1)*0xD1B54A32D192ED03ULL; for(int w=0;w<20;w++) rng_next(&st);
    double sd=1.0/sqrt((double)N);
    for(int i=0;i<N;i++){ J[(size_t)i*N+i]=0; for(int j=i+1;j<N;j++){ float v=(float)(sd*rng_gauss(&st)); J[(size_t)i*N+j]=v; J[(size_t)j*N+i]=v; } }
    for(int c=0;c<2*K;c++){ float*sc=s+(size_t)c*N; for(int i=0;i<N;i++) sc[i]=(rng_u(&st)<0.5)?1.f:-1.f;
      fields(N,J,sc,h+(size_t)c*N); E[c]=energy(N,sc,h+(size_t)c*N); }
    fwrite(J,sizeof(float),(size_t)N*N,fJ);
    for(int r=0;r<2;r++) for(int k=0;k<K;k++) idx[r*K+k]=r*K+k;
    int m=0, total=neq+nmeas;
    for(int t=0;t<total;t++){
      for(int r=0;r<2;r++) for(int k=0;k<K;k++){
        int c=idx[r*K+k]; float*sc=s+(size_t)c*N,*hc=h+(size_t)c*N; double b=beta[k],dEtot=0;
        for(int i=0;i<N;i++){
          double dE=2.0*sc[i]*hc[i];
          if(dE<=0.0 || rng_u(&st)<exp(-b*dE)){
            sc[i]=-sc[i]; float f=2.f*sc[i]; const float*Ji=J+(size_t)i*N;
            for(int j=0;j<N;j++) hc[j]+=f*Ji[j];
            dEtot+=dE; }
        }
        E[c]+=dEtot;
      }
      if(t%500==499) for(int c=0;c<2*K;c++){ fields(N,J,s+(size_t)c*N,h+(size_t)c*N); E[c]=energy(N,s+(size_t)c*N,h+(size_t)c*N); }
      for(int r=0;r<2;r++) for(int k=t%2;k+1<K;k+=2){   // replica-exchange moves
        int c1=idx[r*K+k],c2=idx[r*K+k+1]; double d=(beta[k]-beta[k+1])*(E[c1]-E[c2]);
        int acc=(d>=0.0||rng_u(&st)<exp(d));
        if(acc){ idx[r*K+k]=c2; idx[r*K+k+1]=c1; swap_acc[k]+=1; }
      }
      if(t>=neq && (t-neq)%every==every-1 && m<M){
        for(int k=0;k<K;k++){
          int ca=idx[k],cb=idx[K+k]; const float*sa=s+(size_t)ca*N,*sb=s+(size_t)cb*N; double ov=0;
          for(int i=0;i<N;i++) ov+=sa[i]*sb[i];
          size_t o=((size_t)smp*K+k)*M+m;
          q[o]=(float)(ov/N); ea[o]=(float)(energy(N,sa,h+(size_t)ca*N)/N);
          int8_t*dst=buf+(((size_t)k*M+m)*2)*N;
          for(int i=0;i<N;i++){ dst[i]=(int8_t)sa[i]; dst[N+i]=(int8_t)sb[i]; }
        }
        m++;
      }
    }
    fwrite(buf,1,(size_t)K*M*2*N,fS);
  }
  fclose(fJ); fclose(fS);
  FILE*f;
  snprintf(fn,512,"%s_q.bin",prefix); f=fopen(fn,"wb"); fwrite(q,sizeof(float),(size_t)S*K*M,f); fclose(f);
  snprintf(fn,512,"%s_e.bin",prefix); f=fopen(fn,"wb"); fwrite(ea,sizeof(float),(size_t)S*K*M,f); fclose(f);
  snprintf(fn,512,"%s_T.txt",prefix); f=fopen(fn,"w");
  for(int k=0;k<K;k++) fprintf(f,"%.8f %.4f\n",T[k],k+1<K? swap_acc[k]/((double)S*(neq+nmeas)):0.0);
  fclose(f);
  return 0;
}
'''


# ----------------------------- data -----------------------------
def generate_all():
    """Compile the generator and run it for every size that has no data yet (in parallel)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists("./sk_gen"):                               # compiled into the Colab working directory
        with open("sk_gen.c", "w") as f:
            f.write(C_SOURCE)
        subprocess.run(["gcc", "-O3", "-march=native", "-ffast-math", "-o", "sk_gen", "sk_gen.c", "-lm"], check=True)
    procs = []
    for i, N in enumerate(SIZES):
        if os.path.exists(f"{DATA_DIR}/N{N}.done"):
            continue
        cmd = ["./sk_gen", str(N), str(SAMPLES[N]), str(K), str(TMIN), str(TMAX),
               str(N_EQ[N]), str(N_MEAS), str(EVERY), f"{DATA_DIR}/N{N}", str(100 + i)]
        print("running:", " ".join(cmd), flush=True)
        procs.append((N, subprocess.Popen(cmd)))
    for N, p in procs:
        if p.wait() != 0:
            raise RuntimeError("sk_gen failed")
        open(f"{DATA_DIR}/N{N}.done", "w").close()


def load(N):
    S, M = SAMPLES[N], N_MEAS // EVERY
    C = M // SEQ_LEN
    f = lambda name, dt: np.fromfile(f"{DATA_DIR}/N{N}_{name}.bin", dtype=dt)
    Tfile = np.loadtxt(f"{DATA_DIR}/N{N}_T.txt")
    J = f("J", np.float32).reshape(S, N, N)
    spins = f("spins", np.int8).reshape(S, K, M, 2, N)
    spins = spins[:, :, :C * SEQ_LEN].reshape(S, K, C, SEQ_LEN, 2, N)
    perm = np.random.default_rng(SPLIT_SEED + N).permutation(S)          # fixed split by disorder sample
    n_test = int(round(TEST_FRACTION * S))
    return dict(N=N, T=Tfile[:, 0], swap=Tfile[:-1, 1], C=C,
                q=f("q", np.float32).reshape(S, K, M).astype(np.float64),
                e=f("e", np.float32).reshape(S, K, M).astype(np.float64),
                J=torch.from_numpy(J).to(device),
                spins=torch.from_numpy(np.ascontiguousarray(spins)).to(device),   # int8, (S,K,C,L,2,N)
                test=np.sort(perm[:n_test]), train=np.sort(perm[n_test:]))


def equilibration_report(data):
    """(1) <q^2> in the first and second half of the measurements. (2) For Gaussian couplings the identity
    e = -(1 - [<q^2>]) / (2T) holds in equilibrium up to O(1/N). Differences are quoted with standard errors."""
    T = data[SIZES[0]]["T"]
    rep = {}
    print("\nequilibration checks (difference, standard error in brackets)")
    for N in SIZES:
        d = data[N]; S = len(d["q"]); q2 = d["q"] ** 2; h = q2.shape[2] // 2
        dh = q2[:, :, :h].mean(2) - q2[:, :, h:].mean(2)
        ident = d["e"].mean(2) + (1.0 - q2.mean(2)) / (2.0 * T[None, :])
        rep[str(N)] = {"swap_min": float(d["swap"].min()), "swap_mean": float(d["swap"].mean()), "rows": []}
        print(f"  N={N}: swap acceptance min {d['swap'].min():.2f}, mean {d['swap'].mean():.2f}")
        for k in (0, 2, 4, 7, 11):
            row = [float(T[k]), float(dh[:, k].mean()), float(dh[:, k].std() / S ** 0.5),
                   float(ident[:, k].mean()), float(ident[:, k].std() / S ** 0.5)]
            rep[str(N)]["rows"].append(row)
            print(f"     T={row[0]:.3f}   <q^2> half1-half2 = {row[1]:+.4f} ({row[2]:.4f})"
                  f"    e + (1-<q^2>)/(2T) = {row[3]:+.4f} ({row[4]:.4f})")
    return rep


# ----------------------------- model (unchanged) -----------------------------
class SKGNNLSTM(nn.Module):
    """Message-passing GNN on the complete graph (edge weights J_ij), followed by an LSTM over time.

    mode = "overlap": node feature q_i = s_i^a s_i^b, with the global sign fixed so that q >= 0.
                      The rescaled overlap |q| N^(1/3) is fed to the LSTM alongside the pooled GNN embedding.
    mode = "single" : node feature s_i^a of one replica only.
    """
    def __init__(self, mode, use_gnn, rescale):
        super().__init__()
        assert mode in ("overlap", "single") and (use_gnn or mode == "overlap")
        self.mode, self.use_gnn, self.rescale = mode, use_gnn, rescale
        if use_gnn:
            self.embed = nn.Linear(1, HIDDEN)
            self.msg = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(GNN_LAYERS)])
            self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(),
                                                    nn.Linear(HIDDEN, HIDDEN)) for _ in range(GNN_LAYERS)])
        in_dim = (1 if mode == "overlap" else 0) + (HIDDEN if use_gnn else 0)
        self.lstm = nn.LSTM(in_dim, LSTM_HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN), nn.ReLU(), nn.Linear(LSTM_HIDDEN, 1))

    def forward(self, spins, J):
        # spins: (B, L, 2, N) float in {-1,+1};  J: (B, N, N)
        N = spins.shape[-1]
        feats = []
        if self.mode == "overlap":
            qi = spins[:, :, 0] * spins[:, :, 1]                      # (B, L, N) local overlap
            q = qi.mean(-1, keepdim=True)                             # (B, L, 1)
            sgn = torch.sign(q)
            sgn = sgn + (sgn == 0).to(sgn.dtype)                      # treat q = 0 as positive
            node = qi * sgn                                           # remove the global Z2 sign
            scale = N ** (1.0 / 3.0) if self.rescale else 1.0
            feats.append(q.abs() * scale)
        else:
            node = spins[:, :, 0]                                     # (B, L, N)
        if self.use_gnn:
            h = self.embed(node.unsqueeze(-1))                        # (B, L, N, H)
            Jb = J.unsqueeze(1)                                       # (B, 1, N, N)
            for msg, upd in zip(self.msg, self.upd):
                agg = torch.matmul(Jb, msg(h))                        # sum_j J_ij W h_j -> (B, L, N, H)
                h = h + upd(torch.cat([h, agg], dim=-1))
            feats.append(h.mean(dim=2))                               # permutation-invariant pooling
        x = torch.cat(feats, dim=-1)                                  # (B, L, in_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)                      # logit, (B,)


# ----------------------------- training and prediction (unchanged except for the seed) -----------------------------
def batch_inputs(d, s, k, c):
    x = d["spins"][s, k, c].float()                                   # (B, L, 2, N)
    return x, d["J"][s]


def train(model, data, window, seed):
    low, high = [list(w) for w in window]
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    ex = {}
    for N, d in data.items():
        s, k, c = np.meshgrid(d["train"], low + high, np.arange(d["C"]), indexing="ij")
        y = np.isin(k, low).astype(np.float32)                        # 1 = spin glass, 0 = paramagnet
        ex[N] = [torch.from_numpy(a.ravel()).to(device) for a in (s, k, c, y)]
    rng = np.random.default_rng(seed)                                 # batch order depends on the seed
    model.train()
    for epoch in range(EPOCHS):
        batches = []
        for N in ex:
            p = rng.permutation(len(ex[N][0]))
            batches += [(N, p[i:i + BATCH]) for i in range(0, len(p), BATCH)]
        for j in rng.permutation(len(batches)):
            N, idx = batches[j]
            idx = torch.from_numpy(idx).to(device)
            s, k, c, y = [a[idx] for a in ex[N]]
            x, J = batch_inputs(data[N], s, k, c)
            flip = (torch.randint(0, 2, (len(idx), 1, 1, 1), device=device) * 2 - 1).float()
            loss = loss_fn(model(x * flip, J), y)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def predict(model, d):
    """Mean P(spin glass) for every held-out disorder sample and temperature: array (n_test, K)."""
    model.eval()
    test, C = d["test"], d["C"]
    P = np.zeros((len(test), K))
    for k in range(K):
        for i in range(0, len(test), 16):
            smp = torch.from_numpy(test[i:i + 16]).to(device)
            s = smp.repeat_interleave(C)
            c = torch.arange(C, device=device).repeat(len(smp))
            x, J = batch_inputs(d, s, torch.full_like(s, k), c)
            p = torch.sigmoid(model(x, J)).view(len(smp), C).mean(1)
            P[i:i + len(smp), k] = p.cpu().numpy()
    return P


def train_predict(name, window, seed, data):
    """Train one experiment for one window and one seed. Returns {N: P array (n_test, K)}."""
    torch.manual_seed(seed)                                           # network initialisation depends on the seed
    mode, use_gnn, rescale = EXPERIMENTS[name]
    model = SKGNNLSTM(mode, use_gnn, rescale).to(device)
    train(model, data, window, seed)
    return {N: predict(model, data[N]) for N in SIZES}


def spins_only_accuracy(window, seed, data):
    """Control: a classifier that sees one spin configuration of one replica and no couplings.
    The gauge argument says such a classifier cannot beat 50% on held-out disorder samples."""
    torch.manual_seed(seed)
    low, high = [list(w) for w in window]
    out = {}
    for N in SIZES:
        d = data[N]

        def make(samples):
            s, k, c = np.meshgrid(samples, low + high, np.arange(d["C"]), indexing="ij")
            y = torch.from_numpy(np.isin(k, low).astype(np.float32).ravel()).to(device)
            s, k, c = [torch.from_numpy(a.ravel()).to(device) for a in (s, k, c)]
            x = d["spins"][s, k, c][:, 0, 0].float()                  # first snapshot of each sequence, replica a: (n, N)
            return x, y

        xtr, ytr = make(d["train"])
        xte, yte = make(d["test"])
        model = nn.Sequential(nn.Linear(N, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        for epoch in range(EPOCHS):
            perm = torch.from_numpy(rng.permutation(len(xtr))).to(device)
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                loss = loss_fn(model(xtr[b]).squeeze(-1), ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (model(xte).squeeze(-1) > 0).float()
            out[str(N)] = float((pred == yte).float().mean().item())
    return out


# ----------------------------- T_c estimators -----------------------------
def crossing(T, y, lo, hi, ref=None, target=0.5):
    """Temperature in [lo, hi] where y changes sign. If there are several: with `ref`, take the one where ref is
    closest to `target` (discards noise crossings in the flat tails); without `ref`, the steepest one."""
    best, score = None, 1e9
    for k in range(len(T) - 1):
        if y[k] * y[k + 1] < 0:
            f = y[k] / (y[k] - y[k + 1])
            t = T[k] + (T[k + 1] - T[k]) * f
            if ref is None:
                sc = -abs((y[k + 1] - y[k]) / (T[k + 1] - T[k]))
            else:
                sc = abs(ref[k] + f * (ref[k + 1] - ref[k]) - target)
            if lo <= t <= hi and sc < score:
                best, score = t, sc
    return best


def bootstrap(fn, sizes_n, rng):
    """fn(list of resampled index arrays) -> crossing or None. Returns (mean, std), or NaN if rarely found."""
    vals = [fn([rng.integers(0, n, n) for n in sizes_n]) for _ in range(N_BOOT)]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if len(vals) > N_BOOT / 2 else (float("nan"), float("nan"))


def extrapolate(tstar, err):
    """Weighted fit T*(N) = T_inf + a N^(-1/3), the finite-size form for the SK model. Returns (T_inf, error)."""
    x = np.array(SIZES, float) ** (-1.0 / 3.0)
    w = 1.0 / np.maximum(np.array(err), 1e-3) ** 2
    A = np.vstack([np.ones(len(x)), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ A.T @ (w * np.array(tstar))
    return float(coef[0]), float(cov[0, 0] ** 0.5)


def analyse(P, T, window, rng):
    """All estimates for one trained experiment (one window, one seed)."""
    low, high = [list(x) for x in window]
    pairs = list(zip(SIZES[:-1], SIZES[1:]))
    out = {}
    out["acc"] = float(np.mean([np.mean(np.concatenate([P[N][:, low].ravel() > 0.5, P[N][:, high].ravel() < 0.5]))
                                for N in SIZES]))
    half = {N: bootstrap(lambda ix: crossing(T, P[N][ix[0]].mean(0) - 0.5, LEVEL_RANGE[0], LEVEL_RANGE[1]),
                         [len(P[N])], rng) for N in SIZES}
    out["half"] = {str(N): list(half[N]) for N in SIZES}
    if all(np.isfinite(half[N][0]) for N in SIZES):
        out["T_inf"] = list(extrapolate([half[N][0] for N in SIZES], [half[N][1] for N in SIZES]))
    else:
        out["T_inf"] = [float("nan"), float("nan")]

    def cross(a, b):
        def fn(ix):
            pa, pb = P[a][ix[0]].mean(0), P[b][ix[1]].mean(0)
            return crossing(T, pa - pb, CROSS_RANGE[0], CROSS_RANGE[1], (pa + pb) / 2, 0.5)
        return bootstrap(fn, [len(P[a]), len(P[b])], rng)
    out["cross"] = {f"{a}-{b}": list(cross(a, b)) for a, b in pairs}
    out["P"] = {str(N): P[N].mean(0).tolist() for N in SIZES}
    return out


# ----------------------------- aggregation over seeds -----------------------------
def agg(pairs_list):
    v = np.array([p[0] for p in pairs_list], float)
    e = np.array([p[1] for p in pairs_list], float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return dict(mean=float("nan"), seed_std=float("nan"), boot=float("nan"), n=0)
    return dict(mean=float(v[ok].mean()), seed_std=float(v[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
                boot=float(e[ok].mean()), n=int(ok.sum()))


def show(a, n_total):
    if a["n"] == 0:
        return "not found".ljust(30)
    return f"{a['mean']:.3f} +- {a['seed_std']:.3f} ({a['boot']:.3f}, {a['n']}/{n_total})"


def summarize(results):
    seeds = [str(s) for s in SEEDS if str(s) in results["runs"]]
    n = len(seeds)
    pairs = [f"{a}-{b}" for a, b in zip(SIZES[:-1], SIZES[1:])]
    summary = {}
    print("\n" + "=" * 100)
    print(f"SUMMARY over {n} seeds {seeds}.   Format:  mean +- spread over seeds  (mean bootstrap error, seeds found / total)")
    print(f"Exact: T_c = {TC_REF}")
    print("=" * 100)
    for name in EXPERIMENTS:
        print(f"\n--- {name} ---")
        summary[name] = {}
        for w in WINDOWS:
            runs = [results["runs"][s][name][w] for s in seeds]
            S = {"T_inf": agg([r["T_inf"] for r in runs]),
                 "half": {str(N): agg([r["half"][str(N)] for r in runs]) for N in SIZES},
                 "cross": {p: agg([r["cross"][p] for r in runs]) for p in pairs},
                 "acc": float(np.mean([r["acc"] for r in runs]))}
            print(f"  window {w}    accuracy {S['acc']:.3f}")
            print("     level crossing T*(N):   " + "   ".join(f"N={N}: {show(S['half'][str(N)], n)}" for N in SIZES))
            print(f"     extrapolated T_inf:     {show(S['T_inf'], n)}")
            print("     size crossing:          " + "   ".join(f"({p}): {show(S['cross'][p], n)}" for p in pairs))
            summary[name][w] = S
    print("\n--- spins_only (one configuration, no couplings): held-out accuracy, chance = 0.500 ---")
    summary["spins_only"] = {}
    for w in WINDOWS:
        acc = {str(N): [results["runs"][s]["spins_only"][w][str(N)] for s in seeds] for N in SIZES}
        summary["spins_only"][w] = {N: [float(np.mean(v)), float(np.std(v, ddof=1)) if n > 1 else float("nan")]
                                    for N, v in acc.items()}
        print(f"  window {w}:  " + "   ".join(f"N={N}: {np.mean(v):.3f} +- {(np.std(v, ddof=1) if n > 1 else float('nan')):.3f}"
                                               for N, v in acc.items()))
    results["summary"] = summary
    return summary


# ----------------------------- main -----------------------------
def main():
    generate_all()
    data = {N: load(N) for N in SIZES}
    T = data[SIZES[0]]["T"]
    pairs = list(zip(SIZES[:-1], SIZES[1:]))
    config = dict(SIZES=SIZES, SAMPLES={str(k): v for k, v in SAMPLES.items()}, N_EQ={str(k): v for k, v in N_EQ.items()},
                  K=K, N_MEAS=N_MEAS, EVERY=EVERY, SEQ_LEN=SEQ_LEN, EPOCHS=EPOCHS, BATCH=BATCH, HIDDEN=HIDDEN,
                  TEST_FRACTION=TEST_FRACTION, SPLIT_SEED=SPLIT_SEED, CROSS_RANGE=list(CROSS_RANGE),
                  WINDOWS={w: [list(a), list(b)] for w, (a, b) in WINDOWS.items()})
    print(f"device: {device}    seeds: {SEEDS}    exact T_c = {TC_REF}")
    for w, (a, b) in WINDOWS.items():
        print(f"window {w}: low T in [{T[a[0]]:.2f},{T[a[-1]]:.2f}], high T in [{T[b[0]]:.2f},{T[b[-1]]:.2f}]")

    results = {"config": config, "T": T.tolist(), "runs": {}}
    if os.path.exists(RESULTS_FILE):
        old = json.load(open(RESULTS_FILE))
        if old.get("config") == config:
            results = old
            print(f"resuming: seeds already finished: {sorted(results['runs'], key=int)}")
        else:
            print(f"{RESULTS_FILE} was made with different settings and will be overwritten")

    results["equilibration"] = equilibration_report(data)

    # conventional baselines from the overlap (all disorder samples; do not depend on the seed)
    rng = np.random.default_rng(12345)
    q2 = {N: (data[N]["q"] ** 2).mean(2) for N in SIZES}
    q4 = {N: (data[N]["q"] ** 4).mean(2) for N in SIZES}
    g = lambda N, ix: 0.5 * (3.0 - q4[N][ix].mean(0) / q2[N][ix].mean(0) ** 2)
    base = {"q2_rescaled": {}, "binder": {}}
    print("\nbaselines, crossings between successive sizes:")
    for a, b in pairs:
        n = [len(q2[a]), len(q2[b])]
        r1 = bootstrap(lambda ix: crossing(T, a ** (2 / 3) * q2[a][ix[0]].mean(0) - b ** (2 / 3) * q2[b][ix[1]].mean(0),
                                           BASE_RANGE[0], BASE_RANGE[1]), n, rng)
        r2 = bootstrap(lambda ix: crossing(T, g(a, ix[0]) - g(b, ix[1]), BASE_RANGE[0], BASE_RANGE[1]), n, rng)
        base["q2_rescaled"][f"{a}-{b}"], base["binder"][f"{a}-{b}"] = list(r1), list(r2)
        print(f"   ({a},{b}):  N^(2/3)<q^2> {r1[0]:.3f}({r1[1]:.3f})    Binder parameter {r2[0]:.3f}({r2[1]:.3f})")
    results["baselines"] = base

    for seed in SEEDS:
        if str(seed) in results["runs"]:
            continue
        t0 = time.time()
        rng = np.random.default_rng(1000 + seed)                         # bootstrap resampling
        run = {"spins_only": {}}
        for name in EXPERIMENTS:
            run[name] = {}
            for w, window in WINDOWS.items():
                P = train_predict(name, window, seed, data)
                r = analyse(P, T, window, rng)
                run[name][w] = r
                print(f"seed {seed}  {name:17s} window {w}:  T_inf = {r['T_inf'][0]:.3f}   size crossings "
                      + " ".join(f"{v[0]:.3f}" for v in r["cross"].values()), flush=True)
        for w, window in WINDOWS.items():
            run["spins_only"][w] = spins_only_accuracy(window, seed, data)
            print(f"seed {seed}  spins_only        window {w}:  accuracy "
                  + " ".join(f"{v:.3f}" for v in run["spins_only"][w].values()), flush=True)
        results["runs"][str(seed)] = run
        json.dump(results, open(RESULTS_FILE, "w"))                      # checkpoint after every seed
        print(f"seed {seed} finished in {(time.time() - t0) / 60:.1f} min and saved to {RESULTS_FILE}\n", flush=True)

    summarize(results)
    json.dump(results, open(RESULTS_FILE, "w"))

    # figure: mean over seeds (line) and min-max over seeds (band)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = [str(s) for s in SEEDS if str(s) in results["runs"]]
    fig, axes = plt.subplots(len(EXPERIMENTS), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 3.2 * len(EXPERIMENTS)),
                             squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(SIZES)))
    for i, name in enumerate(EXPERIMENTS):
        for j, w in enumerate(WINDOWS):
            ax = axes[i, j]
            for col, N in zip(colors, SIZES):
                curves = np.array([results["runs"][s][name][w]["P"][str(N)] for s in seeds])
                ax.fill_between(T, curves.min(0), curves.max(0), color=col, alpha=0.25, lw=0)
                ax.plot(T, curves.mean(0), "-", color=col, lw=1.4, label=f"N={N}")
            ax.axvline(TC_REF, color="k", ls="--", lw=1)
            ax.axhline(0.5, color="gray", lw=0.6)
            ax.set_title(f"{name}, window {w}", fontsize=9)
            ax.set_xlabel("T"); ax.set_ylabel("P(spin glass)"); ax.set_ylim(-0.03, 1.03)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=180)
    print(f"\nsaved {RESULTS_FILE} and {FIGURE_FILE}")


main()