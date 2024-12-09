def pytest_configure(config):
    config.addinivalue_line(
        "markers", "custom_marker: Description of what this marker is used for."
    )


