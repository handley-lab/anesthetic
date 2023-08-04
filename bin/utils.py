from packaging import version
import subprocess


def run(*args):
    """Run a bash command and return the output in Python."""
    return subprocess.run(args, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def unit_incremented(a, b):
    """Check if a is one version larger than b."""
    a = version.parse(a)
    b = version.parse(b)
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
