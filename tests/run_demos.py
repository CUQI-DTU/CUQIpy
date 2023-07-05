# -*- coding: utf-8 -*-
# Add basic tests here.
#%%
from pathlib import Path

demos_passed = {}

def test_demos():
    here = Path(__file__).parent

    demos = [x.resolve() for x in (here.parent / "demos").glob("*.py")]

    for demo in demos:
        if not demos_passed.get(demo, False):
            print(f"Running demo: {demo}")
            with demo.open("r") as fh:
                exec(fh.read())
            demos_passed[demo] = True

# %%
test_demos()

# %%
