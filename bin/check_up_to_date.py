import pytest
import requests
from packaging import version
import importlib
from latest_dependencies import deps


packages = [importlib.import_module(package) for package in deps]


@pytest.mark.parametrize('package', packages)
def test_packages(package):
    response = requests.get(f"https://pypi.org/pypi/{package.__name__}/json")
    latest_version = response.json()['info']['version']

    if version.parse(latest_version) > version.parse(package.__version__):
        print(f"You should upgrade the {package.__name__} requirement "
              f"from {package.__version__} to {latest_version}")
    assert version.parse(latest_version) <= version.parse(package.__version__)
