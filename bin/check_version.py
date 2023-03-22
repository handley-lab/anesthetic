#!/usr/bin/env python
import sys
import subprocess
from packaging import version
from utils import unit_incremented
vfile = "anesthetic/_version.py"
README = "README.rst"


def run(*args):
    return subprocess.run(args, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


current_version = run("cat", vfile)
current_version = current_version.split("=")[-1].strip().strip("'")

run("git", "fetch", "origin", "master")
previous_version = run("git", "show", "remotes/origin/master:" + vfile)
previous_version = previous_version.split("=")[-1].strip().strip("'")

readme_version = run("grep", ":Version:", README)
readme_version = readme_version.split(":")[-1].strip()

if version.parse(current_version) != version.parse(readme_version):
    sys.stderr.write("Version mismatch: {} != {}".format(vfile, README))
    sys.exit(1)

elif not unit_incremented(current_version, previous_version):
    sys.stderr.write(("Version must be incremented by one:\n"
                      "HEAD:   {},\n"
                      "master: {}.\n"
                      ).format(current_version, previous_version))
    sys.exit(1)
