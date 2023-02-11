#!/usr/bin/env python
import sys
from packaging import version
import subprocess
vfile = "anesthetic/_version.py"
README = "README.rst"


def run(*args):
    return subprocess.run(args, stdout=subprocess.PIPE, text=True).stdout

current_version = run("cat", vfile)
current_version = current_version.split("=")[-1].strip().strip("'")
current_version = version.parse(current_version)

#previous_version = run("git", "show", "master:" + vfile)
#previous_version = previous_version.split("=")[-1].strip().strip("'")
#previous_version = version.parse(previous_version)

readme_version = run("grep", ":Version:", README)
readme_version = readme_version.split(":")[-1].strip()
readme_version = version.parse(readme_version)

if current_version != readme_version:
    sys.stderr.write("Version mismatch: {} != {}".format(vfile, README))
    sys.exit(1)

#elif current_version <= previous_version:
#    sys.stderr.write("Version must be incremented: {} <= {}".format(vfile, "master:" + vfile))
#    sys.exit(1)
