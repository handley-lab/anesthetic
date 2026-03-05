#!/usr/bin/env python
import tomli

with open("pyproject.toml", 'rb') as f:
    pyproject = tomli.load(f)

deps = pyproject["project"]["dependencies"]
deps = [dep.partition("==")[0] for dep in deps]
deps = [dep.partition(">=")[0] for dep in deps]
deps = [dep.partition("<=")[0] for dep in deps]
deps = [dep.partition(">")[0] for dep in deps]
deps = [dep.partition("<")[0] for dep in deps]
deps = [dep.partition("~=")[0] for dep in deps]
deps = [dep.partition("^=")[0] for dep in deps]

if __name__ == "__main__":
    deps = [f'"{dep}"' for dep in deps]
    print(' '.join(deps))
