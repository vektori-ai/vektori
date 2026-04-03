"""Unit tests for quality filter."""

from vektori.config import QualityConfig
from vektori.ingestion.filter import is_quality_sentence


def test_good_sentence_passes():
    text = "I prefer WhatsApp for all my communications with the bank."
    assert is_quality_sentence(text) is True


def test_substantive_sentence_passes():
    text = "My outstanding loan amount is forty-five thousand rupees as of this month."
    assert is_quality_sentence(text) is True


def test_junk_ok_filtered():
    assert is_quality_sentence("ok") is False


def test_junk_yeah_filtered():
    assert is_quality_sentence("yeah sure") is False


def test_junk_hi_filtered():
    assert is_quality_sentence("hi there") is False


def test_too_short_filtered():
    assert is_quality_sentence("yes") is False


def test_filter_disabled():
    config = QualityConfig(enabled=False)
    assert is_quality_sentence("ok", config) is True
    assert is_quality_sentence("hi", config) is True


def test_code_filtered():
    assert is_quality_sentence("import os; os.system('rm -rf /')") is False


def test_base64_filtered():
    long_b64 = "dGhpcyBpcyBhIHZlcnkgbG9uZyBiYXNlNjQgc3RyaW5nIHRoYXQgc2hvdWxkIGJlIGZpbHRlcmVk"
    assert is_quality_sentence(long_b64) is False


def test_low_content_density_filtered():
    # Almost all stopwords
    assert is_quality_sentence("it is what it is and that is that for us") is False


def test_min_chars_configurable():
    config = QualityConfig(min_chars=5, min_words=2)
    assert is_quality_sentence("I pay monthly.", config) is True
