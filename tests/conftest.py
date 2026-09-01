import pytest


def pytest_addoption(parser):
    parser.addoption('--exhaustive', action='store_true', default=False,
                     help='also run tests marked "exhaustive" (large seed/size sweeps, several minutes)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--exhaustive'):
        return
    skip_exhaustive = pytest.mark.skip(reason='exhaustive sweep, use --exhaustive to run')
    for item in items:
        if 'exhaustive' in item.keywords:
            item.add_marker(skip_exhaustive)
