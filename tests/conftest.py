import pytest
from pathlib import Path
from shutil import copy


@pytest.fixture
def copy_reference(tmp_path):
    HERE = Path(__file__).parent

    def _copy_reference(fname):
        src = HERE / fname
        dest = tmp_path / src.stem
        copy(src, dest)

        return dest.resolve()

    return _copy_reference
