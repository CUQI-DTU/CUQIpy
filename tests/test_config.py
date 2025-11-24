import cuqi

def test_progress_bar_update_frequently_default():
    """Config exposes PROGRESS_BAR_UPDATE_FREQUENTLY and it is a boolean."""
    assert hasattr(cuqi, "config")
    assert isinstance(cuqi.config.PROGRESS_BAR_UPDATE_FREQUENTLY, bool)

def test_progress_bar_update_frequently_toggle(monkeypatch):
    """Can toggle PROGRESS_BAR_UPDATE_FREQUENTLY via monkeypatch."""
    original = cuqi.config.PROGRESS_BAR_UPDATE_FREQUENTLY
    monkeypatch.setattr(cuqi.config, "PROGRESS_BAR_UPDATE_FREQUENTLY", not original, raising=True)
    assert cuqi.config.PROGRESS_BAR_UPDATE_FREQUENTLY == (not original)
    # monkeypatch will restore the original value after the test