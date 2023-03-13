#!/usr/bin/env python
import sys
from packaging import version
import subprocess
vfile = "anesthetic/_version.py"
README = "README.rst"


def run(*args):
    return subprocess.run(args, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


current_version = run("cat", vfile)
current_version = current_version.split("=")[-1].strip().strip("'")
current_version = version.parse(current_version)

run("git", "fetch", "origin", "master")
previous_version = run("git", "show", "remotes/origin/master:" + vfile)
previous_version = previous_version.split("=")[-1].strip().strip("'")
previous_version = version.parse(previous_version)

readme_version = run("grep", ":Version:", README)
readme_version = readme_version.split(":")[-1].strip()
readme_version = version.parse(readme_version)

HEAD = run("git", "rev-parse", "HEAD")

version.parse('2.0.0c0').pre

a = readme_version
a.base_version
def unit_incremented(a, b):
    """Check if a is one version larger than b."""
    if a.pre is not None and b.pre is not None:
        if a.pre[0] == b.pre[0]:
            return a.pre[1] == b.pre[1]+1 and a.base_version == b.base_version
        else:
            return (a.pre[1] == 0 and a.pre[0] > b.pre[0] 
                    and a.base_version == b.base_version)
    elif a.pre is not None:
        return a.base_version > b.base_version and a.pre[1] == 0
    elif b.pre is not None:
        return a.base_version == b.base_version
    else:
        return (a.micro == b.micro+1 and 
                a.minor == b.minor and 
                a.major == b.major or
                a.micro == 0 and
                a.minor == b.minor+1 and 
                a.major == b.major or 
                a.micro == 0 and
                a.minor == 0 and 
                a.major == b.major+1)


for a, b in [['2.0.0b2', '2.0.0b1'],
             ['2.0.0b0', '2.0.0a3'],
             ['2.0.0', '2.0.0b1'],
             ['3.0.0a0', '2.5.6'],
             ['2.0.3', '2.0.2'], 
             ['2.1.0', '2.0.5'],
             ['3.0.0', '2.5.6'],
             ]:
    assert unit_incremented(version.parse(a), version.parse(b))


for a, b in [['2.0.0b3', '2.0.0b1'],
             ['2.0.0b3', '2.0.0a3'],
             ['2.0.1', '2.0.0b1'],
             ['3.0.0a1', '2.5.6'],
             ['2.0.4', '2.0.2'], 
             ['2.1.5', '2.0.5'],
             ['3.5.6', '2.5.6'],
             ]:
    assert not unit_incremented(version.parse(a), version.parse(b))


if current_version != readme_version:
    sys.stderr.write("Version mismatch: {} != {}".format(vfile, README))
    sys.exit(1)

elif unit_incremented(current_version, previous_version):
    sys.stderr.write(("Version must be incremented by one:\n"
                      "HEAD:   {},\n"
                      "master: {}.\n"
                      ).format(current_version, previous_version))
    sys.exit(1)
