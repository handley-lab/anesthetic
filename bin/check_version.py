#!/usr/bin/env python
import sys
from utils import readme_version, unit_incremented, run

README = "README.rst"

with open(README) as f:
    current_version = readme_version(f)

run("git", "fetch", "origin", "master")
previous_version = readme_version(
    run("git", "show", "remotes/origin/master:" + README).splitlines())

if not unit_incremented(current_version, previous_version):
    sys.stderr.write(("Version must be incremented by one:\n"
                      "HEAD:   {},\n"
                      "master: {}.\n"
                      ).format(current_version, previous_version))
    sys.exit(1)
