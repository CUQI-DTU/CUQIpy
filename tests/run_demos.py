# -*- coding: utf-8 -*-
# Add basic tests here.

from pathlib import Path


def test_demos():
    here = Path(__file__).parent

    demos = [x.resolve() for x in (here.parent / "demos").glob("*.py")]

    for demo in demos:
        print(f"Running demo: {demo}")
        with demo.open("r") as fh:
            exec(fh.read())
