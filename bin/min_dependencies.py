#!/usr/bin/env python
import tomli

with open("pyproject.toml", 'rb') as f:
    pyproject = tomli.load(f)

deps = pyproject["project"]["dependencies"]
deps = [dep.replace(">=", "==") for dep in deps]
deps = [dep.replace("~=", "==") for dep in deps]
deps = [dep[:dep.find('<')-1] if '<' in dep else dep for dep in deps]

print(' '.join(deps))
