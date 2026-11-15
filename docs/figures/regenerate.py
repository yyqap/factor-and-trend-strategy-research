"""Regenerate figures from public aggregate results and declared configuration.

Run from the repository root with matplotlib and pandas installed:
    python docs/figures/regenerate.py
"""
from pathlib import Path
import os
import json
os.environ.setdefault('MPLCONFIGDIR', '/tmp/factor-trend-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
COLORS={'ridge':'#777777','logistic':'#0072B2','lightgbm':'#009E73'}
NAMES={'ridge':'Ridge','logistic':'Logistic','lightgbm':'LightGBM'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})

comparison=pd.read_csv(ROOT/'results/task1/model_comparison.csv').set_index('model')
nav=pd.read_csv(ROOT/'results/task1/monthly_baseline_nav.csv',parse_dates=['date'])
models=['ridge','logistic','lightgbm'];x=np.arange(3);width=.24
fig,axes=plt.subplots(1,2,figsize=(12.8,5.1))
for offset,field,label,color in [(-1,'cv_mean_rank_ic','Development CV','#777777'),
    (0,'historical_full_universe_rank_ic','Historical full universe','#0072B2'),
    (1,'industry_adjusted_rank_ic','Industry-adjusted historical','#009E73')]:
    axes[0].bar(x+offset*width,[comparison.loc[m,field] for m in models],width,label=label,color=color)
axes[0].set_xticks(x,[NAMES[m] for m in models]);axes[0].set_ylabel('Mean monthly Rank IC (correlation)')
axes[0].set_title('Ranking diagnostics: different periods / samples',loc='left');axes[0].legend(frameon=False,fontsize=8)
axes[0].set_ylim(0,.17)
for m in models:
    g=nav[nav.model.eq(m)&nav.cost_bps.eq(10)].sort_values('date')
    axes[1].plot(g.date,g.nav,label=NAMES[m],color=COLORS[m],lw=1.8)
axes[1].set_title('Frozen Top30 equal-weight baseline',loc='left');axes[1].set_ylabel('Reconstructed absolute NAV (initial = 1)')
axes[1].set_xlabel('Monthly holding-exit observation date');axes[1].legend(frameon=False)
axes[1].xaxis.set_major_locator(mdates.YearLocator());axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Task 1 | Model comparison and reconstructed baseline NAV',x=.07,ha='left',weight='bold',fontsize=14)
fig.text(.07,.025,'Source: checked-in aggregate research exports. CV: 66 development months; historical signals: Jun 2014–Jun 2018 (49 months).\n'
    'NAV: Jul 1, 2014–Aug 1, 2018; 10 bps per traded side; monthly observations only. Industry IC uses published-as-of CSRC labels, minimum 5 stocks/group.',fontsize=8,color='#444444')
fig.subplots_adjust(top=.85,bottom=.21,left=.07,right=.98,wspace=.3)
fig.savefig(OUT/'task1_model_comparison.png',dpi=180);plt.close(fig)

catalog=pd.read_csv(ROOT/'results/task1/portfolio_metrics.csv');catalog=catalog[catalog.cost_bps.eq(10)]
markers={'stock_book':'o','market_overlay':'s','relative_state':'^'}
groups={'stock_book':'Stock count / weights','market_overlay':'Market / cash overlays','relative_state':'Relative-state controls'}
fig,axes=plt.subplots(1,3,figsize=(12.8,5.1),sharex=True,sharey=True)
for ax,m in zip(axes,models):
    g=catalog[catalog.model.eq(m)]
    for group,marker in markers.items():
        h=g[g.catalog_group.eq(group)]
        ax.scatter(h.monthly_observation_nav_mdd*100,h.cagr*100,s=45,marker=marker,color=COLORS[m],alpha=.7,label=groups[group])
    b=g[g.catalog_group.eq('stock_book')&g.policy_id.eq('top30_equal')].iloc[0]
    ax.scatter([b.monthly_observation_nav_mdd*100],[b.cagr*100],s=155,facecolors='none',edgecolors='#222222',lw=1.3)
    ax.annotate('Top30 baseline',(b.monthly_observation_nav_mdd*100,b.cagr*100),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.set_title(f'{NAMES[m]}: 16 evaluated designs',loc='left');ax.grid(alpha=.2)
axes[0].set_ylabel('CAGR, actual elapsed dates (%)');axes[2].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('Task 1 | All 48 evaluated portfolio designs at 10 bps',x=.07,ha='left',weight='bold',fontsize=14)
fig.text(.525,.115,'Monthly-observation NAV MDD (%)',ha='center',fontsize=10)
fig.text(.07,.025,'Source: results/task1/portfolio_metrics.csv; Jul 1, 2014–Aug 1, 2018. Open circles identify the common Top30 baseline, not a test-selected optimum.\n'
    'The catalog contains executed experiments only; it is not a Cartesian search of all combinations. Monthly-observation MDD excludes unknown intramonth drawdowns.',fontsize=8,color='#444444')
fig.subplots_adjust(top=.85,bottom=.21,left=.07,right=.98,wspace=.12)
fig.savefig(OUT/'task1_portfolio_tradeoffs.png',dpi=180);plt.close(fig)

t2=pd.read_csv(ROOT/'results/task2/strategy_nav.csv',parse_dates=['date'])
fig,axes=plt.subplots(2,1,figsize=(12.8,6.1),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
axes[0].plot(t2.date,t2.nav,color='#0072B2',lw=1.7,label='Net native-data research strategy')
axes[0].set_ylabel('Strategy NAV (initial = 1)');axes[0].legend(frameon=False,loc='upper left')
axes[1].fill_between(t2.date,-t2.drawdown*100,0,color='#0072B2',alpha=.18)
axes[1].plot(t2.date,-t2.drawdown*100,color='#0072B2',lw=1)
axes[1].set_ylabel('Daily observed NAV drawdown (%)');axes[1].set_xlabel('Joint valuation date')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('Task 2 | Full native-underlying run, net of stated research costs',x=.08,ha='left',weight='bold',fontsize=14)
fig.text(.08,.885,'16 logical assets / 20 underlying legs | First execution Jun 4, 2020 | Cutoff Aug 18, 2023',fontsize=10)
fig.text(.08,.025,'Source: checked-in strategy-only aggregate NAV; initial observation Jun 3, 2020. CAGR 2.54%; daily Sharpe (RF=0, sqrt252) 0.94; daily MDD 3.44%.\n'
    '5 bps per one-way traded notional plus 0.5% annual management fee. Fractional positions, zero-interest CNY cash and synchronous EOD sizing are research assumptions.\n'
    'No exact daily reference-index NAV is available. This is a prototype with explicit roll / basket / FX conventions, not certified replication or live performance.',fontsize=8,color='#444444')
fig.subplots_adjust(top=.83,bottom=.19,left=.08,right=.98,hspace=.18)
fig.savefig(OUT/'task2_native_nav.png',dpi=180);plt.close(fig)

# Homepage overview: point estimates only, with different samples separated.
# Exact historical dates and complete NAV/risk figures remain in the results guide.
fig,axes=plt.subplots(1,2,figsize=(12.8,4.9),sharey=True)
overview=[
    (axes[0], [('cv_mean_rank_ic','Development CV','#777777'),
               ('historical_full_universe_rank_ic','Later full-universe evaluation','#0072B2')],
     'Model selection and historical evaluation\n66 validation months / 49 evaluation months'),
    (axes[1], [('industry_raw_matched_rank_ic','Raw IC on the identical retained stocks','#777777'),
               ('industry_adjusted_rank_ic','Two-sided industry-adjusted IC','#009E73')],
     'Industry check on an identical sample\n49 months; known CSRC groups with at least 5 stocks')]
for ax,fields,title in overview:
    for shift,(field,label,color) in zip([-.17,.17],fields):
        values=[comparison.loc[m,field] for m in models]
        bars=ax.bar(x+shift,values,.31,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=8.5)
    ax.set_xticks(x,[NAMES[m] for m in models])
    ax.set_title(title,loc='left',fontsize=10.5,pad=12)
    ax.set_ylim(0,.19);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=8.1,loc='upper left')
axes[0].set_ylabel('Mean monthly Spearman Rank IC (correlation)')
fig.suptitle('Task 1 | Ranking evidence across three model families',x=.07,ha='left',weight='bold',fontsize=14)
fig.text(.07,.025,'Source: results/task1/model_comparison.csv. Historical-data research, not a synthetic demo; IC is not accuracy or investment return.\n'
    'Point estimates shown. Different periods / samples are separated above; detailed dates, uncertainty and complete backtests are retained in docs/results_guide.md.',
    fontsize=8,color='#444444')
fig.subplots_adjust(top=.81,bottom=.20,left=.07,right=.98,wspace=.18)
fig.savefig(OUT/'task1_ranking_evidence.png',dpi=180);plt.close(fig)

# Homepage architecture: declared rules and account identities, without P&L
# promises or generated demonstration prices. Counts come from public metadata.
universe=pd.read_csv(ROOT/'results/task2/universe.csv')
logical_count=len(universe)
leg_count=len({leg for value in universe.actual_legs for leg in value.split(';')})
config=json.loads((ROOT/'configs/task2.json').read_text())
rules=config['rules']
assert logical_count==16 and leg_count==20 and rules['reversal_confirmation']=='price_above_ma5'
fig,ax=plt.subplots(figsize=(12.8,6.0))
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
xs=[.03,.355,.68];top=.55;bottom=.17;bw=.29;bh=.25
nodes=[
    (xs[0],top,'1  Inputs and instrument identity',
     f'{logical_count} logical exposures / {leg_count} accounting legs\n5 funded indices + 15 futures legs\nObserved sessions and trading masks'),
    (xs[1],top,'2  Lagged trend signals',
     'Native-observation lookbacks\nFour-component percentile ranking\nDecisions use previously known signals'),
    (xs[2],top,'3  Portfolio rules',
     f'Top{rules["top_n"]} selection; volatility-aware sizing\nPortfolio volatility bands / exposure scaling\nAsset caps and residual cash\nMA5-confirmed recovery allocation boost'),
    (xs[2],bottom,'4  Staggered execution',
     f'Monthly common sessions {rules["rebalance_days"][0]}–{rules["rebalance_days"][-1]}\nPhased target changes; closed-market deferral\nNew positions earn from the next interval'),
    (xs[1],bottom,'5  Native P&L and costs',
     'Funded principal vs futures variation margin\nSame-contract returns, rolls and FX marks\n'
     f'{rules["transaction_cost"]*10000:g} bps per side + {rules["management_fee"]*100:g}% annual management'),
    (xs[0],bottom,'6  Reconcile and evaluate',
     'Cash + funded market value = NAV\nP&L less trading and management fees\nNet daily NAV and risk diagnostics')]
for left,low,title,body in nodes:
    ax.add_patch(FancyBboxPatch((left,low),bw,bh,boxstyle='round,pad=0.008,rounding_size=0.008',
                 linewidth=.9,edgecolor='#B7BDC3',facecolor='#FFFFFF'))
    ax.text(left+.016,low+bh-.038,title,ha='left',va='top',fontsize=10.3,weight='bold',color='#0072B2')
    ax.text(left+.016,low+bh-.102,body,ha='left',va='top',fontsize=9.05,linespacing=1.8,color='#222222')
arrows=[((xs[0]+bw+.008,top+bh/2),(xs[1]-.008,top+bh/2)),
        ((xs[1]+bw+.008,top+bh/2),(xs[2]-.008,top+bh/2)),
        ((xs[2]+bw/2,top-.012),(xs[2]+bw/2,bottom+bh+.012)),
        ((xs[2]-.008,bottom+bh/2),(xs[1]+bw+.008,bottom+bh/2)),
        ((xs[1]-.008,bottom+bh/2),(xs[0]+bw+.008,bottom+bh/2))]
for start,end in arrows:
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#666666'))
ax.text(.03,.963,'Task 2 | From native observations to a reconciled research account',ha='left',va='top',fontsize=14,weight='bold')
ax.text(.03,.888,'Fixed rules, explicit instrument accounting and testable execution timing',ha='left',va='top',fontsize=10.5,color='#444444')
ax.plot([.03,.97],[.10,.10],lw=.7,color='#B7BDC3')
ax.text(.03,.07,'Source: configs/task2.json, results/task2/universe.csv and the public Task2 signal / rule / account modules.\n'
    'Research assumptions include fractional positions, zero-interest CNY cash and synchronous end-of-day sizing.\n'
    'Historical backtests and generated-data demos are labeled separately; neither is a live execution record.',
    ha='left',va='top',fontsize=8.2,linespacing=1.4,color='#444444')
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
fig.savefig(OUT/'task2_research_workflow.png',dpi=180);plt.close(fig)
print('Wrote 5 figures from public aggregate results, configuration and declared research rules.')
