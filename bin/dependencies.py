#!/usr/bin/env python
import tomli

with open("pyproject.toml", 'rb') as f:
    pyproject = tomli.load(f)

print(' '.join(pyproject["project"]["dependencies"]))
