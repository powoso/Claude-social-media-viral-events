"""Tests for data collector utilities (no network calls)."""

from __future__ import annotations

import pytest

from social_media_markets.collectors.socialblade import SocialBladeCollector
from social_media_markets.collectors.app_store import AppStoreCollector


class TestSocialBladeParser:
    def test_parse_number_plain(self):
        assert SocialBladeCollector._parse_number("1234567") == 1234567
        assert SocialBladeCollector._parse_number("1,234,567") == 1234567

    def test_parse_number_suffixes(self):
        assert SocialBladeCollector._parse_number("1.5M") == 1500000
        assert SocialBladeCollector._parse_number("2.3K") == 2300
        assert SocialBladeCollector._parse_number("1B") == 1000000000

    def test_parse_number_edge_cases(self):
        assert SocialBladeCollector._parse_number("--") is None
        assert SocialBladeCollector._parse_number("N/A") is None
        assert SocialBladeCollector._parse_number("") is None
        assert SocialBladeCollector._parse_number("+500") == 500

    def test_parse_date_formats(self):
        from datetime import datetime, timezone

        dt = SocialBladeCollector._parse_sb_date("Jan 15, 2024")
        assert dt is not None
        assert dt.month == 1
        assert dt.day == 15
        assert dt.year == 2024

        dt = SocialBladeCollector._parse_sb_date("2024-01-15")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_invalid_date(self):
        assert SocialBladeCollector._parse_sb_date("not a date") is None


class TestAppStoreDownloadEstimate:
    def test_rank_1_estimate(self):
        est = AppStoreCollector.estimate_daily_downloads(1)
        assert est > 100000  # top app gets 100k+ downloads

    def test_rank_100_estimate(self):
        est = AppStoreCollector.estimate_daily_downloads(100)
        assert 1000 < est < 50000

    def test_rank_1000_estimate(self):
        est = AppStoreCollector.estimate_daily_downloads(1000)
        assert 100 < est < 5000

    def test_monotonic_decrease(self):
        """Higher rank should mean fewer downloads."""
        d1 = AppStoreCollector.estimate_daily_downloads(1)
        d10 = AppStoreCollector.estimate_daily_downloads(10)
        d100 = AppStoreCollector.estimate_daily_downloads(100)
        assert d1 > d10 > d100
