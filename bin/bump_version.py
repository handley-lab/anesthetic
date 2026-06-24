#!/usr/bin/env python
from utils import readme_version, run
from packaging import version
import sys

README = "README.rst"

with open(README) as f:
    current_version = readme_version(f)
escaped_version = current_version.replace(".", r"\.")
current_version = version.parse(current_version)

if len(sys.argv) > 1:
    update_type = sys.argv[1]
else:
    update_type = "micro"

major = current_version.major
minor = current_version.minor
micro = current_version.micro

if update_type == "micro":
    micro += 1
elif update_type == "minor":
    minor += 1
    micro = 0
elif update_type == "major":
    major += 1
    minor = 0
    micro = 0

new_version = version.parse(f"{major}.{minor}.{micro}")

if sys.platform == "darwin":  # macOS sed requires empty string for backup
    run("sed", "-i", "", f"s/{escaped_version}/{new_version}/g", README)
else:
    run("sed", "-i", f"s/{escaped_version}/{new_version}/g", README)

run("git", "add", README)
run("git", "commit", "-m", f"bump version to {new_version}")
