import pytest

@pytest.fixture(autouse=True)
def example_request_inspection(request):
    print(f"Test Name: {request.node.name}")
    print(f"Markers: {request.node.own_markers}")
    print(f"Function Object: {request.function}")

@pytest.mark.custom_marker
def test_example():
    assert 1 + 1 == 2
    print("test done")

