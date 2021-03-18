import cuqi

def test_Normal():
    assert cuqi.distribution.Normal(0,1).mean == 0