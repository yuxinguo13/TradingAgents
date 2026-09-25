"""Chinese names: the override must win, and a guess must never look curated.

The failure this file guards is not "a name is missing" — it is a *wrong* name
printed with the same authority as a right one. A mechanical gloss off an
English suffix is useful; a mechanical gloss the reader cannot tell apart from a
hand-checked name is a liability.
"""

import json

import pytest

from tradingagents.live import zhnames
from tradingagents.live.zhnames import DERIVED_MARK, ZhNames, derive


@pytest.mark.unit
class TestResolution:
    def test_the_readers_own_file_wins_over_the_curated_table(self):
        r = ZhNames(overrides={"NVDA": "老黄的公司"})
        got = r.get("NVDA", "NVIDIA Corporation")
        assert got.zh == "老黄的公司" and got.source == "override"

    def test_the_curated_table_wins_over_the_mechanical_gloss(self):
        got = ZhNames(overrides={}).get("NVDA", "NVIDIA Corporation")
        assert got.zh == "英伟达" and got.source == "curated" and not got.derived

    def test_a_derived_name_is_marked_so_it_cannot_pass_as_curated(self):
        got = ZhNames(overrides={}).get("ZZZZ", "Foobar Therapeutics, Inc.")
        assert got.source == "derived" and got.zh == "Foobar制药"
        assert got.label().endswith(DERIVED_MARK)
        assert not ZhNames(overrides={}).get("NVDA", "NVIDIA").label().endswith(DERIVED_MARK)

    def test_an_unknown_name_falls_back_to_english_not_to_the_ticker(self):
        """Printing the ticker twice tells the reader nothing they did not have."""
        got = ZhNames(overrides={}).get("WXYZ", "Random Widgets Inc.")
        assert got.zh == "" and got.label() == "Random Widgets"

    def test_a_symbol_with_no_name_at_all_falls_back_to_the_symbol(self):
        assert ZhNames(overrides={}).get("WXYZ").label() == "WXYZ"

    def test_full_prints_both_names_when_both_are_known(self):
        got = ZhNames(overrides={}).get("NVDA", "NVIDIA Corporation")
        assert got.full() == "英伟达（NVIDIA）"

    def test_lookup_is_case_insensitive_on_the_ticker(self):
        assert ZhNames(overrides={"nvda": "老黄"}).get("NVDA").zh == "老黄"


@pytest.mark.unit
class TestDerive:
    @pytest.mark.parametrize("english,expected", [
        ("Nurix Therapeutics, Inc.", "Nurix制药"),
        ("ATAI Life Sciences N.V.", "ATAI生命科学"),
        ("IDEAYA Biosciences, Inc.", "IDEAYA生物科技"),
        ("Star Bulk Carriers Corp.", "Star Bulk海运"),
        ("BJ's Restaurants, Inc.", "BJ's餐饮"),
        ("Foobar Bancorp", "Foobar银行"),
    ])
    def test_the_descriptive_suffix_is_translated_and_the_proper_noun_is_not(
            self, english, expected):
        """Transliterating the proper noun would invent a name nobody uses."""
        assert derive(english) == expected

    def test_the_longest_phrase_wins(self):
        """"Sciences" inside "Life Sciences" glossed the wrong thing."""
        assert derive("ATAI Life Sciences") == "ATAI生命科学"

    def test_a_name_with_no_recognisable_suffix_returns_nothing(self):
        assert derive("Random Widgets") == ""
        assert derive("") == ""

    def test_legal_forms_are_stripped_before_matching(self):
        assert derive("Kymera Therapeutics, Inc.") == derive("Kymera Therapeutics")


@pytest.mark.unit
class TestOverrideFile:
    def test_a_malformed_override_file_costs_the_override_not_the_run(self, tmp_path):
        path = tmp_path / "company_names_zh.json"
        path.write_text("{not json", encoding="utf-8")
        r = ZhNames(path=path)
        assert r.overrides == {} and r.get("NVDA", "NVIDIA").zh == "英伟达"

    def test_a_missing_file_is_an_ordinary_state(self, tmp_path):
        r = ZhNames(path=tmp_path / "nope.json")
        assert r.overrides == {}

    def test_the_file_is_read_and_upper_cased(self, tmp_path):
        path = tmp_path / "zh.json"
        path.write_text(json.dumps({"aapl": "苹果公司"}), encoding="utf-8")
        assert ZhNames(path=path).get("AAPL").zh == "苹果公司"

    def test_the_path_follows_the_home_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        monkeypatch.delenv("TRADINGAGENTS_ZH_NAMES_PATH", raising=False)
        assert zhnames.override_path() == tmp_path / "company_names_zh.json"
