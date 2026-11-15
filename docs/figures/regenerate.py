"""Regenerate publication figures from the checked-in aggregate CSVs only.

Run from the repository root with matplotlib and pandas installed:
    python docs/figures/regenerate.py
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/factor-trend-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
print('Wrote 3 figures from aggregate CSVs only.')
