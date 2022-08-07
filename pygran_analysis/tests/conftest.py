import pytest


def pytest_addoption(parser):
    parser.addoption("--trajf", action="store", help="trajectory filename")
    parser.addoption("--meshf", action="store", help="mesh filename")


@pytest.fixture
def trajf(request):
    return request.config.getoption("--trajf")


@pytest.fixture
def meshf(request):
    return request.config.getoption("--meshf")
