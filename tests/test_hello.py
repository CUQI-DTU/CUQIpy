import pytest


@pytest.mark.parametrize("greeting", ["hello", "HeLLo", "HeLlO",])
def test_hello(greeting):
    assert greeting.casefold() == "hello".casefold()
