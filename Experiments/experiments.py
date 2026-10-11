# -*- coding: utf-8 -*-
"""
Voyage Control: reproducible experiments and figures (seeded, CPU only).
"""
import json, os, warnings
import numpy as np
from scipy.stats import norm, wilcoxon, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ output dirs
OUT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT, 'figures')
RES_DIR = os.path.join(OUT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(RES_DIR, 'results.json')

# ------------------------------------------------------------------ constants
N_SEEDS = 20
LANES = ['Colombo Express', 'Port Klang Link', 'Singapore Transship', 'Jebel Ali Gulf']
SHORT = ['Colombo', 'Port Klang', 'Singapore', 'Jebel Ali']
BASE = np.array([19.0, 20.5, 21.5, 18.5])          # nominal sailing days
PLAN = BASE.copy()                                   # contracted planned sailing days
COST = np.array([1.00, 1.04, 1.08, 1.15])
G = np.array([[1.00, 1.00, 1.00, 1.00, 1.10, 1.35, 1.50, 1.50, 1.40, 1.10, 1.30, 1.45],   # congestion index by month
              [1.30, 1.25, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.10],
              [1.50, 1.40, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05],
              [1.00, 1.00, 1.00, 1.00, 1.05, 1.20, 1.40, 1.40, 1.20, 1.00, 1.00, 1.00]])
K_CONG, SIG0, SIGJ, PS = 11.0, 1.2, 1.5, 0.07
SUP_U = np.clip(np.random.default_rng(12345).beta(8, 2, 14), 0.5, 0.98)   # supplier on-time rates
SUP_SD = np.random.default_rng(54321).uniform(0.5, 3.5, 14)               # supplier sailing-delay spread
COL = ['#0072B2', '#009E73', '#E69F00', '#D55E00']                        # lane colours (Okabe-Ito)

def sigma(route, month):
    s = SIG0 + SIGJ * ((route == 3) & (month >= 6) & (month <= 8))
    return np.sqrt(s ** 2 + (K_CONG * G[route, month - 1] * 0.05) ** 2)

def gen_corpus(seed, n=864, ps=PS):
    r = np.random.default_rng(seed)
    month = r.integers(1, 13, n); route = r.integers(0, 4, n); sup = r.integers(0, 14, n)
    w = r.choice([1, 2, 3], n, p=[.5, .3, .2]); u = SUP_U[sup]
    g_obs = G[route, month - 1] * np.exp(r.normal(0, 0.05, n))
    sg = SIG0 + SIGJ * ((route == 3) & (month >= 6) & (month <= 8))
    mu = BASE[route] + K_CONG * (g_obs - 1) + 0.8 * (w - 1) + 5 * (1 - u)
    stress = r.random(n) < ps
    T = mu + r.normal(0, sg) + stress * r.uniform(3, 9, n)
    return dict(month=month, route=route, sup=sup, w=w, u=u, T=T, y=(T - PLAN[route] > 5).astype(int), stress=stress)

def feats(route, month, u, w):
    oh = np.eye(4)[route]
    return np.column_stack([oh, month, G[route, month - 1], u, w, COST[route]])
FNAMES = ['route_0', 'route_1', 'route_2', 'route_3', 'month', 'congestion', 'supplier', 'weight', 'cost']
GROUPS = {'Lane identity + cost index': [0, 1, 2, 3, 8], 'Congestion index': [5], 'Month': [4],
          'Supplier reliability': [6], 'Weight class': [7]}

def true_stats(route, month, u, w, ps=PS):
    g = G[route, month - 1]
    mu = BASE[route] + K_CONG * (g - 1) + 0.8 * (w - 1) + 5 * (1 - u)
    sg = sigma(route, month); md = mu - PLAN[route]
    sk = np.linspace(3, 9, 13)[:-1] + 0.25
    p = (1 - ps) * (1 - norm.cdf((5 - md) / sg)) + ps * np.mean([1 - norm.cdf((5 - md - s) / sg) for s in sk], axis=0)
    return mu + ps * 6.0, p

def mm(x):
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    return (x - lo) / (hi - lo + 1e-9)

def utility(t, p, c, wt=(0.5, 0.3, 0.2)):
    return wt[0] * mm(t) + wt[1] * mm(p) + wt[2] * mm(c)

def lane_matrix(model_fn, month, u, w):
    """predictions for all 4 candidate lanes for each shipment -> (n,4)"""
    out = []
    for r in range(4):
        X = feats(np.full(len(month), r), month, u, w)
        out.append(model_fn(X))
    return np.column_stack(out)

def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    idx = np.digitize(p, q[1:-1]); e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

# ------------------------------------------------------------------------------ E1/E2/E3: benchmark, decisions, uncertainty
def run_benchmark():
    R = dict(mae={}, auc={}, brier={}, regret={}, top1={}, imp=[], cov={}, width={}, rel=[], ece={}, brier_cal={})
    for k in ['Global mean', 'Lane mean', 'Lane×month mean', 'Ridge', 'GBR']: R['mae'][k] = []
    for k in ['Lane×month rate', 'Logistic', 'RF']: R['auc'][k] = []; R['brier'][k] = []
    for k in ['Random lane', 'Always cheapest', 'Seasonal baseline', 'Ridge + Logistic', 'GBR + RF (ML)']: R['regret'][k] = []; R['top1'][k] = []
    cov_groups = ['All', 'Jebel Ali Jun–Aug', 'Colombo Jun–Sep', 'Other']
    for k in ['Split conformal', 'Adaptive conformal']:
        R['cov'][k] = {g: [] for g in cov_groups}; R['width'][k] = []
    imp = {g: [] for g in GROUPS}
    rel_p, rel_y, rel_c = [], [], []
    for seed in range(N_SEEDS):
        d = gen_corpus(seed); n = len(d['T'])
        X = feats(d['route'], d['month'], d['u'], d['w']); T, y = d['T'], d['y']
        fold_mae = {k: [] for k in R['mae']}; pred_auc = {k: np.zeros(n) for k in R['auc']}
        reg = {k: [] for k in R['regret']}; top1 = {k: [] for k in R['top1']}
        cov = {k: {g: [] for g in cov_groups} for k in R['cov']}; wid = {k: [] for k in R['width']}
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
            rt, mo = d['route'], d['month']
            # --- regression baselines (target statistics fitted on the training fold only)
            gm = T[tr].mean(); lane_m = np.array([T[tr][rt[tr] == r].mean() for r in range(4)])
            lm = np.full((4, 12), np.nan)
            for r in range(4):
                for m in range(1, 13):
                    s = (rt[tr] == r) & (mo[tr] == m)
                    lm[r, m - 1] = T[tr][s].mean() if s.sum() >= 2 else lane_m[r]
            ly = np.full((4, 12), np.nan)
            for r in range(4):
                for m in range(1, 13):
                    s = (rt[tr] == r) & (mo[tr] == m)
                    ly[r, m - 1] = (y[tr][s].sum() + 0.2 * y[tr].mean() * 5) / (s.sum() + 5)
            ridge = Ridge(1.0).fit(X[tr], T[tr]); lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
            gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed).fit(X[tr], T[tr])
            rf = RandomForestClassifier(300, min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
            fold_mae['Global mean'].append(mean_absolute_error(T[te], np.full(len(te), gm)))
            fold_mae['Lane mean'].append(mean_absolute_error(T[te], lane_m[rt[te]]))
            fold_mae['Lane×month mean'].append(mean_absolute_error(T[te], lm[rt[te], mo[te] - 1]))
            fold_mae['Ridge'].append(mean_absolute_error(T[te], ridge.predict(X[te])))
            fold_mae['GBR'].append(mean_absolute_error(T[te], gbr.predict(X[te])))
            pred_auc['Lane×month rate'][te] = ly[rt[te], mo[te] - 1]
            pred_auc['Logistic'][te] = lr.predict_proba(X[te])[:, 1]
            pred_auc['RF'][te] = rf.predict_proba(X[te])[:, 1]
            # --- decision evaluation on test shipments (ground truth from the generator)
            mt, ut, wt_ = d['month'][te], d['u'][te], d['w'][te]
            tt = np.column_stack([true_stats(np.full(len(te), r), mt, ut, wt_)[0] for r in range(4)])
            pp = np.column_stack([true_stats(np.full(len(te), r), mt, ut, wt_)[1] for r in range(4)])
            cc = np.tile(COST, (len(te), 1)); U = utility(tt, pp, cc); orc = U.min(1); best = U.argmin(1)
            def choose(t_hat, p_hat):
                return utility(t_hat, p_hat, cc).argmin(1)
            ch = {}
            ch['Seasonal baseline'] = choose(np.column_stack([lm[r, mt - 1] for r in range(4)]), np.column_stack([ly[r, mt - 1] for r in range(4)]))
            ch['Ridge + Logistic'] = choose(lane_matrix(ridge.predict, mt, ut, wt_), lane_matrix(lambda Z: lr.predict_proba(Z)[:, 1], mt, ut, wt_))
            ch['GBR + RF (ML)'] = choose(lane_matrix(gbr.predict, mt, ut, wt_), lane_matrix(lambda Z: rf.predict_proba(Z)[:, 1], mt, ut, wt_))
            ch['Always cheapest'] = np.zeros(len(te), int)
            for k, c in ch.items():
                reg[k].append((U[np.arange(len(te)), c] - orc).mean()); top1[k].append((c == best).mean())
            reg['Random lane'].append((U.mean(1) - orc).mean()); top1['Random lane'].append(0.25)
            # --- permutation importance (grouped), GBR on held-out fold
            base_mae = fold_mae['GBR'][-1]; rr = np.random.default_rng(seed * 10 + f)
            for g, cols in GROUPS.items():
                v = []
                for _ in range(8):
                    Xp = X[te].copy(); perm = rr.permutation(len(te)); Xp[:, cols] = Xp[perm][:, cols]
                    v.append(mean_absolute_error(T[te], gbr.predict(Xp)) - base_mae)
                imp[g].append(np.mean(v))
            # --- split conformal (marginal) vs normalised/adaptive conformal, alpha = 0.1
            rr2 = np.random.default_rng(seed * 100 + f); perm = rr2.permutation(tr); ntr = int(0.75 * len(perm))
            pr, ca = perm[:ntr], perm[ntr:]
            g1 = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed).fit(X[pr], T[pr])
            s_cal = np.abs(T[ca] - g1.predict(X[ca])); alpha = 0.1
            qk = int(np.ceil((len(ca) + 1) * (1 - alpha))) - 1; q1 = np.sort(s_cal)[min(qk, len(ca) - 1)]
            oof = cross_val_predict(GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed), X[pr], T[pr], cv=3)
            sg = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed).fit(X[pr], np.abs(T[pr] - oof))
            sc = lambda Z: np.maximum(sg.predict(Z), 0.3)
            s2 = s_cal / sc(X[ca]); q2 = np.sort(s2)[min(qk, len(ca) - 1)]
            yh = g1.predict(X[te]); half = {'Split conformal': np.full(len(te), q1), 'Adaptive conformal': q2 * sc(X[te])}
            grp = {'All': np.ones(len(te), bool), 'Jebel Ali Jun–Aug': (rt[te] == 3) & (mo[te] >= 6) & (mo[te] <= 8),
                   'Colombo Jun–Sep': (rt[te] == 0) & (mo[te] >= 6) & (mo[te] <= 9)}
            grp['Other'] = ~(grp['Jebel Ali Jun–Aug'] | grp['Colombo Jun–Sep'])
            for k in half:
                hit = np.abs(T[te] - yh) <= half[k]
                for g_, m in grp.items():
                    if m.sum() > 0: cov[k][g_].append(hit[m])
                wid[k].append(2 * half[k].mean())
            # --- calibration of delay probability (RF trained on 75%, isotonic fitted on calibration part)
            rf2 = RandomForestClassifier(300, min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(X[pr], y[pr])
            iso = IsotonicRegression(out_of_bounds='clip').fit(rf2.predict_proba(X[ca])[:, 1], y[ca])
            pr_raw = rf2.predict_proba(X[te])[:, 1]; rel_p.append(pr_raw); rel_c.append(iso.predict(pr_raw)); rel_y.append(y[te])
        for k in fold_mae: R['mae'][k].append(np.mean(fold_mae[k]))
        for k in pred_auc:
            R['auc'][k].append(roc_auc_score(y, pred_auc[k])); R['brier'][k].append(brier_score_loss(y, np.clip(pred_auc[k], 0, 1)))
        for k in reg: R['regret'][k].append(np.mean(reg[k])); R['top1'][k].append(np.mean(top1[k]))
        for k in cov:
            for g_ in cov_groups: R['cov'][k][g_].append(np.concatenate(cov[k][g_]).mean())
            R['width'][k].append(np.mean(wid[k]))
    R['imp'] = {g: (float(np.mean(v)), float(np.std(v))) for g, v in imp.items()}
    py, pr_, pc_ = np.concatenate(rel_y), np.concatenate(rel_p), np.concatenate(rel_c)
    R['ece'] = {'raw': ece(py, pr_), 'isotonic': ece(py, pc_)}
    R['brier_cal'] = {'raw': brier_score_loss(py, pr_), 'isotonic': brier_score_loss(py, pc_)}
    R['rel'] = (py, pr_, pc_)
    return R

# ------------------------------------------------------------------------------ risk engine
W0 = np.array([0.30, 0.25, 0.20, 0.15, 0.10])   # sailing, handover, inventory, variability, historical
INV_SCORE = np.array([10, 40, 70, 100])

def gen_risk(seed, n, ps, tight=1.0):
    r = np.random.default_rng(seed)
    route = r.integers(0, 4, n); month = r.integers(1, 13, n); sup = r.integers(0, 14, n); w = r.choice([1, 2, 3], n, p=[.5, .3, .2]); u = SUP_U[sup]
    g = G[route, month - 1] * np.exp(r.normal(0, 0.05, n)); sg = SIG0 + SIGJ * ((route == 3) & (month >= 6) & (month <= 8))
    st = r.random(n) < ps
    sail = BASE[route] + K_CONG * (g - 1) + 0.8 * (w - 1) + 5 * (1 - u) + r.normal(0, sg) + st * r.uniform(3, 9, n)
    sd = np.maximum(0, sail - PLAN[route])
    hd = r.gamma(1.5, 1.5, n) + (r.random(n) < ps) * r.uniform(2, 8, n)
    arr = sail + r.uniform(0, 10, n); demand = r.uniform(15, 45, n); safety = demand * r.uniform(5, 10, n)
    stock = demand * r.uniform(40, 80, n) * tight; proj = stock - demand * arr
    tier = np.where(proj < 0, 3, np.where(proj < safety, 2, np.where(proj < 2 * safety, 1, 0)))
    return dict(sd=sd, hd=hd, inv=INV_SCORE[tier], tier=tier, var=SUP_SD[sup], rel=u)

def components(D, sail_sat=12, hand_sat=10):
    S = 100 * np.clip(D['sd'] / sail_sat, 0, 1); H = 100 * np.clip(D['hd'] / hand_sat, 0, 1)
    V = 100 * np.clip(D['var'] / 6.0, 0, 1); Hs = 100 * (1 - D['rel'])
    return np.column_stack([S, H, D['inv'], V, Hs])

def tiers(score, inv_tier):
    t = np.digitize(score, [25, 50, 75]); return np.where(inv_tier == 3, 3, t)

def run_risk():
    out = {}
    # (a) weight perturbation
    D = gen_risk(7, 400, 0.07); Cp = components(D); base = Cp @ W0; bt = tiers(base, D['tier'])
    top = set(np.argsort(-base)[:40]); r = np.random.default_rng(0)
    rows = []
    for alpha in [800, 400, 200, 100, 50, 25, 12]:
        l1, ov, fl, rho = [], [], [], []
        for _ in range(300):
            w = r.dirichlet(alpha * W0); s = Cp @ w
            l1.append(np.abs(w - W0).sum() / 2); ov.append(len(top & set(np.argsort(-s)[:40])) / 40)
            fl.append((tiers(s, D['tier']) != bt).mean()); rho.append(spearmanr(s, base)[0])
        rows.append((alpha, np.mean(l1), np.mean(ov), np.std(ov), np.mean(fl), np.std(fl), np.mean(rho)))
    out['perturb'] = rows
    # (b) saturation-point sweep in a disrupted quarter
    Dd = gen_risk(11, 600, 0.22); ss = np.arange(6, 19, 1); hs = np.arange(6, 17, 1)
    grid = np.zeros((len(hs), len(ss))); redf = (Dd['sd'] > 5).mean()
    for i, h in enumerate(hs):
        for j, s in enumerate(ss):
            sc = components(Dd, s, h) @ W0; grid[i, j] = (tiers(sc, Dd['tier']) >= 2).mean()
    out['sat'] = (ss, hs, grid, redf)
    # (c) stress sweep, quarter of 90 shipments
    rates = np.arange(0, 0.325, 0.025); res = []
    for ps in rates:
        rf_, hr_ = [], []
        for sd_ in range(40):
            Dq = gen_risk(1000 + sd_, 90, ps); sc = components(Dq) @ W0
            rf_.append((Dq['sd'] > 5).mean()); hr_.append((tiers(sc, Dq['tier']) >= 2).mean())
        res.append((ps, np.mean(rf_), 1.96 * np.std(rf_) / np.sqrt(40), np.mean(hr_), 1.96 * np.std(hr_) / np.sqrt(40)))
    out['stress'] = res
    # (d) two operating quarters as in the paper's Table III (fresh seeds, 80 / 90 shipments)
    q = {}
    for name, n, ps in [('Normal quarter', 80, 0.07), ('Disrupted quarter', 90, 0.22)]:
        c = np.zeros(3); t = np.zeros(4); m_ = 0
        for sd_ in range(200):
            Dq = gen_risk(5000 + sd_, n, ps); sc = components(Dq) @ W0; tt = tiers(sc, Dq['tier'])
            c += np.array([((Dq['sd'] <= 2).sum()), ((Dq['sd'] > 2) & (Dq['sd'] <= 5)).sum(), (Dq['sd'] > 5).sum()]); t += np.bincount(tt, minlength=4)
        q[name] = (n, ps, c / 200, t / 200)
    out['quarters'] = q
    return out

# ------------------------------------------------------------------------------ decision map (full-corpus fit)
def decision_map():
    d = gen_corpus(0); X = feats(d['route'], d['month'], d['u'], d['w'])
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, d['T'])
    rf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, d['y'])
    step = 0.02; pts = []
    for a in np.arange(0, 1 + 1e-9, step):
        for b in np.arange(0, 1 - a + 1e-9, step): pts.append((a, b, 1 - a - b))
    pts = np.array(pts); maps = {}
    for mon in (1, 7):
        m = np.array([mon]); u = np.array([0.8]); w = np.array([2])
        t = lane_matrix(gbr.predict, m, u, w); p = lane_matrix(lambda Z: rf.predict_proba(Z)[:, 1], m, u, w)
        tt, pp = mm(t)[0], mm(p)[0]; cc = mm(COST[None, :])[0]
        win = np.array([np.argmin(wt[0] * tt + wt[1] * pp + wt[2] * cc) for wt in pts]); maps[mon] = (win, t[0], p[0])
    cal = {'ml': [], 'oracle': [], 'seasonal': []}
    for mon in range(1, 13):
        m = np.array([mon]); u = np.array([0.8]); w = np.array([2])
        t = lane_matrix(gbr.predict, m, u, w); p = lane_matrix(lambda Z: rf.predict_proba(Z)[:, 1], m, u, w)
        cc = np.tile(COST, (1, 1)); cal['ml'].append(int(utility(t, p, cc).argmin(1)[0]))
        tt = np.column_stack([true_stats(np.array([r]), m, u, w)[0] for r in range(4)]); pp = np.column_stack([true_stats(np.array([r]), m, u, w)[1] for r in range(4)])
        cal['oracle'].append(int(utility(tt, pp, cc).argmin(1)[0]))
    return pts, maps, cal

# ------------------------------------------------------------------------------ main
if __name__ == '__main__':
    import figures
    R = run_benchmark(); RK = run_risk(); pts, maps, cal = decision_map()
    figures.make_all(R, RK, pts, maps, cal, FIG_DIR)
    figures.dump(R, RK, cal, RESULTS_JSON)
    print(f'figures -> {FIG_DIR}')
    print(f'results -> {RESULTS_JSON}')
    print('done')