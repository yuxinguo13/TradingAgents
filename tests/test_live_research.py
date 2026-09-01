"""The links: addresses only, and every one of them well-formed.

This module makes no network call, so the only way it fails is by emitting a URL
that does not resolve — a share class Yahoo hyphenates and everyone else dots, a
company name encoded twice, a query that reads as one long token. Each of those
sends the reader to a blank page from a section whose whole purpose is letting
them check the report against a primary source.
"""

import re
from urllib.parse import parse_qs, urlparse

import pytest

from tradingagents.live import research


@pytest.mark.unit
class TestUrls:
    def test_every_link_is_an_absolute_http_url(self):
        for _, links in research.all_links("NVDA", "NVIDIA"):
            for link in links:
                assert re.match(r"https?://", link.url), link.url
                assert " " not in link.url

    def test_a_share_class_is_written_the_way_each_site_writes_it(self):
        """Yahoo hyphenates BRK-B; Barchart and most others dot it."""
        urls = {l.label: l.url for l in research.quote_links("BRK-B")}
        assert "quote/BRK-B" in urls["Yahoo Finance 行情"]
        assert "BRK.B" in urls["Barchart 观点"]

    def test_the_news_query_is_encoded_once(self):
        """Interpolating a pre-encoded phrase then appending %20stock mixed
        separators, and Google News read the result as a single token."""
        url = next(l.url for l in research.news_links("KNSA", "Kiniksa Pharmaceuticals")
                   if "news.google" in l.url)
        assert "%20" not in url
        assert parse_qs(urlparse(url).query)["q"] == ["Kiniksa Pharmaceuticals stock"]

    def test_a_missing_company_name_falls_back_to_the_ticker(self):
        url = next(l.url for l in research.news_links("KNSA") if "news.google" in l.url)
        assert parse_qs(urlparse(url).query)["q"] == ["KNSA stock"]


@pytest.mark.unit
class TestBlock:
    def test_all_five_tiers_are_rendered(self):
        text = research.markdown_block("NVDA", "NVIDIA")
        for title, _ in research.SECTIONS:
            assert f"**{title}**" in text

    def test_the_chinese_platforms_are_labelled_as_relays(self):
        """They carry the same upstream data; agreement is not confirmation."""
        text = research.markdown_block("NVDA")
        assert "转载源，不作独立验证" in text

    def test_the_thirteen_f_link_carries_its_own_warning(self):
        note = next(l.note for l in research.filing_links("NVDA") if "13F" in l.label)
        assert "滞后" in note and "不能当作买入理由" in note

    def test_the_heading_is_the_callers(self):
        assert research.markdown_block("NVDA", heading="## 去哪查").startswith("## 去哪查")
