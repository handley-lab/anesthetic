#!/usr/bin/env python
import sys
from packaging import version
from utils import get_readme_version, unit_incremented, run

VFILE = "anesthetic/_version.py"
README = "README.rst"

current_version = run("cat", VFILE)
current_version = current_version.split("=")[-1].strip().strip("'")

run("git", "fetch", "origin", "master")
previous_version = run("git", "show", "remotes/origin/master:" + VFILE)
previous_version = previous_version.split("=")[-1].strip().strip("'")

with open(README) as f:
    readme_version = get_readme_version(f)

if version.parse(current_version) != version.parse(readme_version):
    sys.stderr.write("Version mismatch: {} != {}".format(VFILE, README))
    sys.exit(1)

elif not unit_incremented(current_version, previous_version):
    sys.stderr.write(("Version must be incremented by one:\n"
                      "HEAD:   {},\n"
                      "master: {}.\n"
                      ).format(current_version, previous_version))
    sys.exit(1)
