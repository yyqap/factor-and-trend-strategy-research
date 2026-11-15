"""Input-contract and portfolio-rule checks on generated fixtures only."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from factor_trend_research.task2.demo import synthetic_inputs
from factor_trend_research.task2.inputs import FILES, load_native_inputs, validate_inputs
from factor_trend_research.task2.native_strategy import DEFAULT_CONFIG, PreparedSignals, SignalFrames
from factor_trend_research.task2.universe import FUNDED, instruments


def write_fixture(directory, data):
    """An intentionally synthetic loader fixture, not certified native data."""
    directory = Path(directory)
    data.market.reset_index().to_csv(directory / FILES[0], index=False)
    for name, frame in zip(FILES[1:], [data.local_returns, data.risk_returns, data.observed,
                                      data.common_open.to_frame()]):
        frame.to_csv(directory / name, index_label="date")
    specs = []
    for s in instruments():
        specs.append(dict(asset_id=s.asset_id, kind=s.kind, currency=s.currency,
            point_value=s.point_value, representation="price_index" if s.asset_id in FUNDED else "native_futures_contract",
            source="GENERATED_TEST_FIXTURE_NOT_A_PROVIDER", price_type="synthetic", dividend_policy="excluded"))
    meta = {**data.metadata, "input_kind": "native_original_underlyings", "data_gate_passed": True,
            "instruments": specs,
            "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}}
    (directory / "metadata.json").write_text(json.dumps(meta))
    return meta


class NativeInputContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_inputs(periods=360)

    def test_local_loader_roundtrip_preserves_all_twenty_legs(self):
        with tempfile.TemporaryDirectory() as path:
            write_fixture(path, self.data)
            loaded = load_native_inputs(path)
            self.assertEqual(set(loaded.local_returns.columns), set(self.data.local_returns.columns))
            pd.testing.assert_frame_equal(loaded.local_returns, self.data.local_returns, check_freq=False,
                                          rtol=1e-12, atol=1e-12, check_names=False)

    def test_undeclared_etf_proxy_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            meta = write_fixture(path, self.data)
            meta["instruments"][-1]["representation"] = "ETF_proxy"
            (Path(path) / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "substitution"):
                load_native_inputs(path)

    def test_changed_input_bytes_fail_the_passed_gate(self):
        with tempfile.TemporaryDirectory() as path:
            write_fixture(path, self.data)
            with (Path(path) / FILES[1]).open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_native_inputs(path)

    def test_missing_gate_is_not_inferred_from_plausible_data(self):
        with tempfile.TemporaryDirectory() as path:
            meta = write_fixture(path, self.data)
            meta["data_gate_passed"] = False
            (Path(path) / "metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "passed data gate"):
                load_native_inputs(path)

    def test_same_contract_return_and_calendar_disagreement_rejected(self):
        bad = self.data.local_returns.copy()
        bad.iloc[100, 6] += 0.1
        with self.assertRaisesRegex(ValueError, "same-held-contract"):
            validate_inputs(replace(self.data, local_returns=bad))
        common = self.data.common_open.copy()
        common.iloc[100] = False
        with self.assertRaisesRegex(ValueError, "calendar disagrees"):
            validate_inputs(replace(self.data, common_open=common))


class PortfolioRuleChecks(unittest.TestCase):
    def prepared(self, reverse=False):
        dates = pd.bdate_range("2020-01-01", periods=270)
        columns = [f"L{i}" for i in range(16)]
        score = pd.DataFrame(np.tile(np.arange(16, 0, -1), (270, 1)), index=dates, columns=columns)
        vol = {n: pd.DataFrame(0.2, index=dates, columns=columns) for n in (21, 22, 65, 130, 252)}
        corr = pd.DataFrame(0.0, index=dates, columns=columns)
        reversal = pd.DataFrame(False, index=dates, columns=columns)
        reversal.iloc[260, 0] = reverse
        risk = pd.DataFrame(0.0, index=dates, columns=columns)
        return PreparedSignals(SignalFrames(score, vol, corr, reversal), risk)

    def test_reversal_boost_uses_donors_without_adding_gross_exposure(self):
        cfg = replace(DEFAULT_CONFIG, monthly_vol_target=False)
        caps = np.full(16, 0.2)
        base, chosen, _ = self.prepared(False).target(260, caps, cfg)
        boosted, _, count = self.prepared(True).target(260, caps, cfg)
        self.assertEqual(chosen.tolist(), list(range(11)))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(boosted[0] - base[0], 0.05)
        self.assertAlmostEqual(boosted.sum(), base.sum())
        self.assertTrue(np.all(boosted <= caps))
        self.assertTrue(np.all(boosted[11:] == 0))

    def test_caps_leave_residual_cash_instead_of_renormalizing_to_one(self):
        cfg = replace(DEFAULT_CONFIG, reversal=False, monthly_vol_target=False)
        w, _, _ = self.prepared().target(260, np.full(16, 0.02), cfg)
        self.assertTrue(np.all(w <= 0.02))
        self.assertAlmostEqual(w.sum(), 0.22)


if __name__ == "__main__":
    unittest.main()
