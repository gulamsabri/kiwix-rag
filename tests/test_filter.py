import pytest
from kiwix_rag.filter import ChunkFilter


@pytest.fixture
def f():
    return ChunkFilter()


def test_clean_educational_text(f):
    assert f.is_clean(
        "To stop severe bleeding, apply firm direct pressure to the wound "
        "using a clean cloth or bandage and maintain pressure for at least "
        "ten minutes without lifting to check the wound."
    )


def test_strong_ad_postpaid(f):
    assert not f.is_clean(
        "POSTPAID. Hallicrafters SX-28A receiver, good condition. "
        "Send $125 postpaid. Write for our complete catalog."
    )


def test_strong_ad_send_dollar(f):
    assert not f.is_clean("Send $15.00 and we will mail your order today.")


def test_moderate_ad_for_sale(f):
    # Single 'for sale' alone is only +1, below threshold of 2
    score, _ = f.score("Collins KWM-2 for sale, asking $800.")
    assert score >= 1


def test_conspiracy_flat_earth(f):
    score, reasons = f.score("The flat earth society insists that NASA lies.")
    assert score >= 1
    assert any("flat earth" in r for r in reasons)


def test_conspiracy_does_not_block_medicine(f):
    # "depopulation" could appear in legitimate population medicine text
    assert f.is_clean(
        "Smallpox depopulation of native communities was catastrophic, "
        "killing an estimated 90 percent of some populations."
    )


def test_score_returns_tuple(f):
    score, reasons = f.score("normal text")
    assert isinstance(score, int)
    assert isinstance(reasons, list)


def test_custom_threshold(f):
    text = "Send $10.00 for our catalog — items for sale below."
    # default threshold 2 — this should fail (for sale + price = >=2)
    assert not f.is_clean(text)
    # threshold 10 — same text passes
    assert f.is_clean(text, threshold=10)
