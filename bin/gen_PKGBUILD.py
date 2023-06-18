import tomli

with open("pyproject.toml", 'rb') as f:
    pyproject = tomli.load(f)

description = pyproject["project"]["description"]
version = open('anesthetic/_version.py','r').readlines()[0].split("=")[1].strip().strip("'")
url = pyproject["project"]["urls"]["Homepage"]
pyproject["project"]["dependencies"]
rel=1


PKGBUILD = """# Maintainer: Will Handley <wh260@cam.ac.uk> (aur.archlinux.org/account/wjhandley)
pkgname=python-anesthetic
_name=${pkgname#python-}
pkgver=%s
pkgrel=%s
pkgdesc="%s"
arch=(any)
url="%s"
license=(MIT)
groups=()
depends=(python-numpy python-matplotlib python-scipy python-pandas)
makedepends=(python-build python-installer)
provides=(anesthetic)
conflicts=()
replaces=()
backup=()
options=(!emptydirs)
install=
source=("https://files.pythonhosted.org/packages/source/${_name::1}/$_name/$_name-$pkgver.tar.gz")
sha256sums=(XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)

build() {
    cd "$srcdir/$_name-$pkgver"
    python -m build --wheel --no-isolation
}

package() {
    cd "$srcdir/$_name-$pkgver"
    python -m installer --destdir="$pkgdir" dist/*.whl
}
""" % (version, rel, description, url)
print(PKGBUILD)
