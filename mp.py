import pytest
import torch.multiprocessing as mp

def spawn_launcher(pid, function, args, kwargs):
    """Helper function to launch the test function in a separate process."""
    return function(*args, **kwargs)

@pytest.fixture(autouse=True)
def spawn_fixture(request):
    """Fixture to run tests in separate processes if marked with @pytest.mark.spawn."""
    if request.node.get_closest_marker("spawn"):
        function = request.function

        def spawned_function(*args, **kwargs):
            mp.spawn(fn=spawn_launcher, args=(function, args, kwargs), nprocs=2, join=True)

        request._pyfuncitem.obj = spawned_function
    yield

# Example test
@pytest.mark.spawn
def test_example():
    assert 1 + 1 == 2


