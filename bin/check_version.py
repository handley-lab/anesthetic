#!/usr/bin/env python
import sys
from packaging import version
import subprocess
vfile = "anesthetic/_version.py"

current_version = subprocess.run(["cat", vfile],
                                 stdout=subprocess.PIPE, text=True)
previous_version = subprocess.run(["git", "show", "master:" + vfile],
                                  stdout=subprocess.PIPE, text=True)

current_version = current_version.stdout.split("=")[-1].strip().strip("'")
current_version = version.parse(current_version)
previous_version = previous_version.stdout.split("=")[-1].strip().strip("'")
previous_version = version.parse(previous_version)

if current_version > previous_version:
    sys.exit(0)
else:
    sys.exit(1)
