"""Native observation counts and future-mutation causality fixtures."""
import unittest
import numpy as np
import pandas as pd
from factor_trend_research.task2.native_signals import build_signals
from factor_trend_research.task2.universe import LOGICAL

def signal_fixture(periods=600):
    dates=pd.bdate_range('2018-01-02',periods=periods)
    actual=list(dict.fromkeys(a for legs in LOGICAL.values() for a in legs))
    rng=np.random.default_rng(7)
    local=pd.DataFrame(rng.normal(.0002,.005,(periods,len(actual))),index=dates,columns=actual)
    local.iloc[0]=np.nan
    return local,local.copy(),pd.DataFrame(True,index=dates,columns=actual)


class NativeSignalBoundaries(unittest.TestCase):
    def test_260_actual_returns_need261_observed_endpoints(self):
        local,risk,observed=signal_fixture(300)
        frames,_,counts,_,_=build_signals(local,risk,observed)
        self.assertEqual(counts.iloc[259].min(),259)
        self.assertTrue(frames.score.iloc[259].isna().all())
        self.assertTrue(frames.score.iloc[260].notna().all())

    def test_future_missing_episode_cannot_erase_earlier_signals(self):
        local,risk,observed=signal_fixture()
        original=build_signals(local,risk,observed)
        local.iloc[300:302,local.columns.get_loc('ZT')]=np.nan
        risk.iloc[300:302,risk.columns.get_loc('ZT')]=np.nan
        observed.iloc[300,observed.columns.get_loc('ZT')]=False
        changed=build_signals(local,risk,observed)
        pd.testing.assert_frame_equal(original[0].score.iloc[:300],changed[0].score.iloc[:300],check_exact=True)
        pd.testing.assert_frame_equal(original[2].iloc[:300],changed[2].iloc[:300],check_exact=True)
        self.assertTrue(changed[0].score.iloc[300:561].isna().all().all())
        self.assertTrue(changed[0].score.iloc[561].notna().all())
        self.assertEqual(changed[2].iloc[300]['US2Y'],0)

    def test_own_observation_masks_control_counts_and_black_requires_all_legs(self):
        local,risk,observed=signal_fixture()
        observed.loc[observed.index[1::2],['ZT','RB']]=False
        local.loc[local.index[1::2],['ZT','RB']]=0
        risk.loc[risk.index[1::2],['ZT','RB']]=0
        _,_,counts,_,_=build_signals(local,risk,observed)
        self.assertEqual(counts.iloc[300]['US2Y'],150)
        self.assertEqual(counts.iloc[300]['FerrousComplex'],150)
        self.assertEqual(counts.iloc[300]['CSI300'],300)


if __name__=='__main__':unittest.main(verbosity=2)
