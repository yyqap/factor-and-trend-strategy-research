"""Local-price trend signals on observed sessions, separate from CNY account P&L.

Source futures prices are turned into a forward same-contract return index for
signals only. Actual portfolio cash and futures quantities are never valued as
this index. No noisy raw cross-contract price gap enters a trend/volatility.
"""
import numpy as np
import pandas as pd
from .native_strategy import SignalFrames
from .universe import LOGICAL


def logical_returns(actual):
    out=pd.DataFrame(index=actual.index)
    for a,legs in LOGICAL.items():
        values=actual[list(legs)]
        out[a]=sum(values[leg]*w for leg,w in legs.items()).where(values.notna().all(axis=1))
    return out


def build_signals(local_actual, risk_actual, observed):
    if not local_actual.index.equals(risk_actual.index) or not local_actual.index.equals(observed.index):
        raise ValueError('Returns and observed-session masks must have the same date index')
    if observed.isna().any().any() or not all(isinstance(v,(bool,np.bool_)) for v in observed.to_numpy().ravel()):
        raise ValueError('Observed-session masks must be explicit booleans, never missing/string values')
    if local_actual.index.has_duplicates or not local_actual.index.is_monotonic_increasing:
        raise ValueError('Input dates must be unique and ascending')
    local=logical_returns(local_actual)
    risk=logical_returns(risk_actual)
    dates,columns=local.index,list(local.columns)
    score_components=[pd.DataFrame(np.nan,index=dates,columns=columns) for _ in range(4)]
    vols={n:pd.DataFrame(np.nan,index=dates,columns=columns) for n in (21,22,65,130,252)}
    corr=pd.DataFrame(np.nan,index=dates,columns=columns)
    reversal=pd.DataFrame(np.nan,index=dates,columns=columns)
    native_count=pd.DataFrame(0,index=dates,columns=columns)
    levels=pd.DataFrame(np.nan,index=dates,columns=columns)
    details=[]
    usd_logical={'Nasdaq100','SP500','US2Y','US5Y','US10Y','Brent'}
    for a,legs in LOGICAL.items():
        daily=local[a]
        observed_native=observed[list(legs)].all(axis=1)
        if np.isinf(daily.to_numpy()).any() or (daily.dropna()<=-1).any():
            raise ValueError('Signal-chain source returns must be finite or explicit missing, and above-100%')
        # Reset CAUSALLY at each missing interval. A later data gap must not
        # erase an earlier valid signal episode. Base1 is merely normalization
        # at a genuinely observed endpoint, never a made-up contract quote.
        chain=pd.Series(np.nan,index=dates)
        episode=pd.Series(np.nan,index=dates)
        value=None;epoch=-1
        for i,rvalue in enumerate(daily.to_numpy()):
            if not np.isfinite(rvalue):
                value=None;epoch+=1
                if observed_native.iloc[i]:
                    value=1.;chain.iloc[i]=value;episode.iloc[i]=epoch
                continue
            if value is None:
                epoch+=1;value=1.
            else:value*=1+rvalue
            chain.iloc[i]=value;episode.iloc[i]=epoch
        # Source keys include explicit missing-return dates. reindex(method=
        # 'ffill') carries an existing source value across OTHER-market dates,
        # including an existing NaN; it does not fill through an active gap or
        # through260 warm-up source observations after a reset.
        source_dates=dates[observed_native.to_numpy()|daily.isna().to_numpy()]
        raw_native=[pd.Series(np.nan,index=source_dates) for _ in range(4)]
        vol_native={n:pd.Series(np.nan,index=source_dates) for n in vols}
        corr_native=pd.Series(np.nan,index=source_dates)
        rev_native=pd.Series(np.nan,index=source_dates)
        count_native=pd.Series(0.,index=source_dates)
        latest_price=pd.Series(dtype=float)
        latest_ready=pd.Series(dtype=bool)
        for ep in episode.dropna().unique():
            native_dates=dates[observed_native.to_numpy()&(episode.to_numpy()==ep)]
            price=chain.reindex(native_dates).dropna()
            if price.empty:continue
            latest_price=price
            r=price.pct_change(fill_method=None)
            count=pd.Series(np.arange(len(price)),index=price.index)
            ready=count>=260;latest_ready=ready
            v={n:r.rolling(n,min_periods=n).std(ddof=1)*np.sqrt(252) for n in vols}
            r1,r12=price/price.shift(21)-1,price/price.shift(252)-1
            lo,hi=price.rolling(252,min_periods=252).min(),price.rolling(252,min_periods=252).max()
            position=((price-lo)/(hi-lo).replace(0,np.nan)).mask(hi==lo,.5)
            raw=[r1/v[21].clip(lower=.005),r12/v[252].clip(lower=.005),r12,position]
            alignment=pd.Series(0.,index=price.index)
            prev=price
            for n in range(2,261):
                current=price.rolling(n,min_periods=n).mean()
                alignment+=(prev>=current).astype(float)
                prev=current
            alignment=(alignment/259).where(price.rolling(260,min_periods=260).count()==260)
            reverse=pd.Series(False,index=price.index)
            armed=False;valley=np.nan
            for d,alignment_value in alignment.items():
                if not np.isfinite(alignment_value):continue
                if alignment_value>.35:
                    armed=True;valley=np.nan
                    continue
                if not armed:continue
                valley=alignment_value if not np.isfinite(valley) else min(valley,alignment_value)
                reverse.loc[d]=valley<=.1 and alignment_value>=valley+.005
            reverse &= price>price.rolling(5,min_periods=5).mean()
            correlation=r.rolling(252,min_periods=130).corr(v[21])
            for k,part in enumerate(raw):raw_native[k].loc[price.index]=part.where(ready)
            for n in vols:vol_native[n].loc[price.index]=v[n].where(ready)
            corr_native.loc[price.index]=correlation.where(ready)
            rev_native.loc[price.index]=reverse.astype(float).where(ready)
            count_native.loc[price.index]=count
        def align(s):return s.reindex(dates,method='ffill')
        for k,part in enumerate(raw_native):score_components[k][a]=align(part)
        for n in vols:vols[n][a]=align(vol_native[n])
        corr[a]=align(corr_native)
        reversal[a]=align(rev_native)
        native_count[a]=align(count_native).fillna(0).astype(int)
        levels[a]=chain
        bad_dates=dates[daily.isna()]
        if len(bad_dates) and (latest_price.empty or bad_dates[-1]>latest_price.index[-1]):
            latest_price=pd.Series(dtype=float);latest_ready=pd.Series(dtype=bool)
        valid_dates=latest_price.index[latest_ready] if len(latest_price) else pd.DatetimeIndex([])
        details.append({'asset':a,'native_signal_calendar':'observed US index subset with this asset native futures marks' if a in usd_logical else 'this asset own observed China sessions',
          'native_marks_after_last_gap':len(latest_price),
          'first_mark_after_last_gap':str(latest_price.index[0].date()) if len(latest_price) else None,
          'native260_returns_ready':str(valid_dates[0].date()) if len(valid_dates) else None,
          'ready':bool(len(valid_dates))})
    finite=np.logical_and.reduce([f.notna().all(axis=1).to_numpy() for f in score_components])
    ranked=[f.rank(axis=1,pct=True,method='average') for f in score_components]
    score=sum(ranked)/4
    score.loc[~finite]=np.nan
    frames=SignalFrames(score=score,vol=vols,correlation=corr,reversal=reversal)
    complete=score.notna().all(axis=1)&reversal.notna().all(axis=1)
    for f in vols.values():complete &=f.notna().all(axis=1)
    complete &=native_count.ge(260).all(axis=1)
    missing=local.isna().any(axis=1)
    last_bad=dates[missing][-1] if missing.any() else dates[0]
    # Preserve the frozen complete-suffix evaluation rule, while the historical
    # frames themselves remain prefix-causal and retain earlier valid episodes.
    primary_eligible=complete&(dates>last_bad)
    summary={'full16_run_performed':False,'logical_assets':columns,'native_signal_details':details,
             'first_all16_native_signal_ready':str(primary_eligible[primary_eligible].index[0].date()) if primary_eligible.any() else None,
             'first_any_complete_signal_episode_ready':str(complete[complete].index[0].date()) if complete.any() else None,
             'last_missing_logical_source_return':str(last_bad.date()),
             'signal_currency':'local currency, same-contract return index rebased separately after each unknown gap; normalized original equity price index returns',
             'risk_currency':'separate CNY-funded-index return or CNY futuresVM per prior marked notional',
             'missing_price_policy':'No missing active source return is imputed; reset causally and require260 valid selected-session returns after each gap. Primary evaluation remains the complete final suffix.',
             'calendar_note':'US index observed sessions are a conservative subset for USD signal/trade schedules. PublishedCMEholiday risk marks are valuation inputs, not counted as proof of tradability.'}
    return frames,risk,native_count,levels,summary
