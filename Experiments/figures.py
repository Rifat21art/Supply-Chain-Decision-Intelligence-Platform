# -*- coding: utf-8 -*-
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

LANES = ['Colombo Express', 'Port Klang Link', 'Singapore Transship', 'Jebel Ali Gulf']
SHORT = ['Colombo', 'Port Klang', 'Singapore', 'Jebel Ali']
COL = ['#0072B2', '#009E73', '#E69F00', '#D55E00']
NAVY, GREY, LGREY, ML = '#1F3A5F', '#6B7280', '#D1D5DB', '#0072B2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'figure.dpi': 120})

def tag(ax, s, **k):
    t = ax.get_title('center'); ax.set_title('', loc='center'); ax.set_title(s + ' ' + t, loc='left', fontsize=7.5, pad=4)

def strip(ax, data_dict, colors, ylabel, order=None, hline=None, ylim=None, rot=20):
    keys = order or list(data_dict)
    for i, k in enumerate(keys):
        v = np.array(data_dict[k]); c = colors[k]
        ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c, lw=0.7), medianprops=dict(color=c, lw=1.2),
                   whiskerprops=dict(color=c, lw=0.6), capprops=dict(color=c, lw=0.6))
        ax.scatter(i + np.random.default_rng(i).uniform(-0.17, 0.17, len(v)), v, s=3.5, color=c, alpha=0.9, lw=0, zorder=3)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=rot, ha='right' if rot else 'center'); ax.set_ylabel(ylabel)
    if hline is not None: ax.axhline(hline, color=GREY, lw=0.6, ls=':')
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis='y', color='#E5E7EB', lw=0.4); ax.set_axisbelow(True)

def fig_benchmark(R, path):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1.15, 0.9, 1.1], wspace=0.42))
    cm = {'Global mean': '#9CA3AF', 'Lane mean': '#9CA3AF', 'Lane×month mean': '#4B5563', 'Ridge': '#7FB2D9', 'GBR': ML}
    strip(axs[0], R['mae'], cm, 'Transit-time MAE (days), 5-fold CV'); axs[0].set_title('Transit time'); tag(axs[0], '(a)')
    cc = {'Lane×month rate': '#4B5563', 'Logistic': '#7FB2D9', 'RF': ML}
    strip(axs[1], R['auc'], cc, 'ROC-AUC, delay > 5 d', rot=20); axs[1].set_title('Delay flag'); tag(axs[1], '(b)')
    rc = {'Random lane': '#9CA3AF', 'Always cheapest': '#9CA3AF', 'Seasonal baseline': '#4B5563', 'Ridge + Logistic': '#7FB2D9', 'GBR + RF (ML)': ML}
    strip(axs[2], R['regret'], rc, 'Mean regret vs. oracle (utility)'); axs[2].set_title('Route decision'); tag(axs[2], '(c)')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

def fig_uncertainty(R, path):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw=dict(width_ratios=[0.9, 1.1, 1.0], wspace=0.5))
    py, pr, pc = R['rel']; ax = axs[0]
    for p, lab, c, m in [(pr, 'RF raw', '#D55E00', 'o'), (pc, 'RF + isotonic', ML, 's')]:
        q = np.quantile(p, np.linspace(0, 1, 11)); q[-1] += 1e-9; idx = np.digitize(p, q[1:-1])
        xs = [p[idx == b].mean() for b in range(10)]; ys = [py[idx == b].mean() for b in range(10)]
        ax.plot(xs, ys, marker=m, ms=3, lw=0.9, color=c, label=lab)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls='--'); ax.set_xlabel('Predicted delay probability'); ax.set_ylabel('Observed frequency')
    ax.legend(frameon=False, loc='upper left', title='ECE %.3f → %.3f' % (R['ece']['raw'], R['ece']['isotonic']), title_fontsize=6.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('Probability calibration'); tag(ax, '(a)')
    ax = axs[1]; groups = ['All', 'Jebel Ali Jun–Aug', 'Colombo Jun–Sep', 'Other']; w = 0.38
    for j, (k, c) in enumerate([('Split conformal', '#9CA3AF'), ('Adaptive conformal', ML)]):
        m = [np.mean(R['cov'][k][g]) for g in groups]; s = [np.std(R['cov'][k][g]) for g in groups]
        ax.bar(np.arange(4) + (j - 0.5) * w, m, w, yerr=s, color=c, error_kw=dict(lw=0.6, capsize=1.5), label=k)
    ax.axhline(0.9, color='#D55E00', lw=0.8, ls='--'); ax.text(-0.45, 0.91, 'nominal', fontsize=5.5, color='#D55E00', ha='left', va='bottom')
    ax.set_xticks(range(4)); ax.set_xticklabels(['All', 'Jebel Ali\nJun–Aug', 'Colombo\nJun–Sep', 'Other'])
    ax.set_ylim(0.6, 1.1); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_ylabel('Empirical coverage'); ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=5.8, handlelength=1, columnspacing=0.8); ax.set_title('Interval coverage'); tag(ax, '(b)')
    ax = axs[2]; names = list(R['imp']); v = np.array([R['imp'][n][0] for n in names]); e = np.array([R['imp'][n][1] for n in names]); o = np.argsort(v)
    ax.barh(range(len(o)), v[o], xerr=e[o], color=ML, error_kw=dict(lw=0.6, capsize=1.5)); ax.set_yticks(range(len(o)))
    ax.set_yticklabels([names[i].replace(' + ', ' +\n') for i in o], fontsize=6); ax.set_xlabel('MAE increase (days)'); ax.set_title('Permutation importance'); tag(ax, '(c)')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

def fig_risk(RK, path):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw=dict(width_ratios=[1, 1.05, 1], wspace=0.55))
    ax = axs[0]; P = np.array(RK['perturb']); o = np.argsort(P[:, 1]); P = P[o]
    ax.errorbar(P[:, 1], P[:, 2], yerr=P[:, 3], marker='o', ms=3, lw=0.9, color=ML, capsize=1.5, label='Top-decile overlap')
    ax.errorbar(P[:, 1], P[:, 4], yerr=P[:, 5], marker='s', ms=3, lw=0.9, color='#D55E00', capsize=1.5, label='Tier-change rate')
    ax.set_xlabel('Weight perturbation, ½‖Δw‖₁'); ax.set_ylabel('Fraction'); ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc='center right')
    ax.set_title('Weight robustness'); tag(ax, '(a)')
    ax = axs[1]; ss, hs, grid, redf = RK['sat']
    im = ax.imshow(grid * 100, origin='lower', aspect='auto', cmap='Blues', extent=[ss[0] - .5, ss[-1] + .5, hs[0] - .5, hs[-1] + .5])
    ax.plot([12], [10], marker='*', ms=7, mfc='#D55E00', mec='white', mew=0.5); ax.set_xlabel('Sailing saturation point (days)'); ax.set_ylabel('Handover saturation (days)')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('High+Critical (%)', fontsize=6.3); cb.ax.tick_params(labelsize=6, width=0.4, length=1.8); cb.outline.set_linewidth(0.4)
    ax.set_title('Saturation-point sensitivity'); tag(ax, '(b)'); ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax = axs[2]; S = np.array(RK['stress']) * 1.0
    ax.errorbar(S[:, 0] * 100, S[:, 1] * 100, yerr=S[:, 2] * 100, marker='o', ms=3, lw=0.9, color='#D55E00', capsize=1.5, label='Red-flagged (sailing > 5 d)')
    ax.errorbar(S[:, 0] * 100, S[:, 3] * 100, yerr=S[:, 4] * 100, marker='s', ms=3, lw=0.9, color=ML, capsize=1.5, label='High or Critical risk')
    ax.axvline(7, color=GREY, lw=0.6, ls=':'); ax.axvline(22, color=GREY, lw=0.6, ls=':')
    ax.set_xlabel('Stress-event rate (%)'); ax.set_ylabel('Share of shipments (%)'); ax.set_ylim(0, 62); ax.legend(frameon=False, loc='upper left', fontsize=5.8); ax.set_title('Stress response'); tag(ax, '(c)')
    ax.grid(color='#E5E7EB', lw=0.4); ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

def fig_decision(pts, maps, cal, path):
    fig = plt.figure(figsize=(7.0, 2.05)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.12)
    h = np.sqrt(3) / 2
    for k, mon in enumerate((1, 7)):
        ax = fig.add_subplot(gs[0, k]); win, t, p = maps[mon]
        x = pts[:, 1] + 0.5 * pts[:, 2]; y = pts[:, 2] * h
        ax.scatter(x, y, c=[COL[i] for i in win], s=14.5, marker='s', lw=0, zorder=2)
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color='#374151', lw=0.7, zorder=3)
        ax.scatter([0.3 + 0.5 * 0.2], [0.2 * h], marker='*', s=40, color='white', edgecolor='black', lw=0.6, zorder=4)
        ax.text(-0.02, -0.07, 'Speed', ha='left', fontsize=6.5, fontweight='bold'); ax.text(1.02, -0.07, 'Reliability', ha='right', fontsize=6.5, fontweight='bold'); ax.text(0.5, h + 0.04, 'Cost', ha='center', fontsize=6.5, fontweight='bold')
        share = [(win == i).mean() * 100 for i in range(4)]
        ax.set_title(['January', 'July'][k] + ' shipment', pad=4); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.12, h + 0.12)
        tag(ax, '(%s)' % 'ab'[k], dx=0.0, dy=0.98)
    handles = [Patch(fc=COL[i], label=SHORT[i]) for i in range(4)] + [Line2D([], [], marker='*', color='w', mfc='white', mec='black', ms=7, label='default weights')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.27, -0.04), ncol=5, frameon=False, fontsize=6, handlelength=1, columnspacing=0.9, handletextpad=0.4)
    ax = fig.add_subplot(gs[0, 2]); mons = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    for r, (lab, key) in enumerate([('Oracle', 'oracle'), ('ML optimiser', 'ml')]):
        for m in range(12):
            ax.add_patch(plt.Rectangle((m, 1 - r), 0.94, 0.9, fc=COL[cal[key][m]], ec='white', lw=0.4))
            if key == 'ml' and cal['ml'][m] != cal['oracle'][m]: ax.text(m + 0.47, 1 - r + 0.45, '×', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    agree = np.mean(np.array(cal['ml']) == np.array(cal['oracle']))
    ax.set_xlim(0, 12); ax.set_ylim(-0.15, 2.05); ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(['ML optimiser', 'Oracle']); ax.set_xticks(np.arange(12) + 0.47); ax.set_xticklabels(mons)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title('Lane chosen at default weights (agree %d/12)' % round(agree * 12)); tag(ax, '(c)', dx=0.0, dy=0.98)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03); plt.close(fig)

def make_all(R, RK, pts, maps, cal, outdir):
    fig_benchmark(R, outdir + '/fig2_benchmark.png'); fig_uncertainty(R, outdir + '/fig3_uncertainty.png')
    fig_risk(RK, outdir + '/fig4_risk.png'); fig_decision(pts, maps, cal, outdir + '/fig5_decision.png')

def ms(v): return (float(np.mean(v)), float(np.std(v)))

def dump(R, RK, cal, path):
    out = {'mae': {k: ms(v) for k, v in R['mae'].items()}, 'auc': {k: ms(v) for k, v in R['auc'].items()},
           'brier': {k: ms(v) for k, v in R['brier'].items()}, 'regret': {k: ms(v) for k, v in R['regret'].items()},
           'top1': {k: ms(v) for k, v in R['top1'].items()}, 'imp': R['imp'], 'ece': R['ece'], 'brier_cal': R['brier_cal'],
           'cov': {k: {g: ms(v) for g, v in d.items()} for k, d in R['cov'].items()}, 'width': {k: ms(v) for k, v in R['width'].items()},
           'cal_ml': cal['ml'], 'cal_oracle': cal['oracle'],
           'perturb': [list(map(float, r)) for r in RK['perturb']], 'stress': [list(map(float, r)) for r in RK['stress']],
           'quarters': {k: [v[0], v[1], v[2].tolist(), v[3].tolist()] for k, v in RK['quarters'].items()},
           'redflag_sat': float(RK['sat'][3]), 'sat_at_12_10': float(RK['sat'][2][list(RK['sat'][1]).index(10), list(RK['sat'][0]).index(12)]),
           'sat_min': float(RK['sat'][2].min()), 'sat_max': float(RK['sat'][2].max())}
    g, b = np.array(R['mae']['GBR']), np.array(R['mae']['Lane×month mean'])
    out['gbr_vs_season_mae'] = {'diff_mean': float((g - b).mean()), 'wins': int((g < b).sum()), 'p': float(wilcoxon(g, b).pvalue)}
    a, c = np.array(R['regret']['GBR + RF (ML)']), np.array(R['regret']['Seasonal baseline'])
    out['ml_vs_season_regret'] = {'diff_mean': float((a - c).mean()), 'wins': int((a < c).sum()), 'p': float(wilcoxon(a, c).pvalue)}
    json.dump(out, open(path, 'w'), indent=1)