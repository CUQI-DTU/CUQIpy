import cuqi

def test_progress_bar_update_frequently_default():
    """Config exposes PROGRESS_BAR_DYNAMIC_UPDATE and it is a boolean."""
    assert hasattr(cuqi, "config")
    assert isinstance(cuqi.config.PROGRESS_BAR_DYNAMIC_UPDATE, bool)

def test_progress_bar_update_frequently_toggle(monkeypatch):
    """Can toggle PROGRESS_BAR_DYNAMIC_UPDATE via monkeypatch."""
    original = cuqi.config.PROGRESS_BAR_DYNAMIC_UPDATE
    monkeypatch.setattr(cuqi.config, "PROGRESS_BAR_DYNAMIC_UPDATE", not original, raising=True)
    assert cuqi.config.PROGRESS_BAR_DYNAMIC_UPDATE == (not original)
    # monkeypatch will restore the original value after the test