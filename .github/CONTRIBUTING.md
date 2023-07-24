Thank you for considering contributing to anesthetic.

If you have a new feature/bug report, make sure you create an [issue](https://github.com/handley-lab/anesthetic/issues), and consult existing ones first, in case your suggestion is already being addressed.

If you want to go ahead and create the feature yourself, you should fork the repository to you own github account and create a new branch with an appropriate name. Commit any code modifications to that branch, push to GitHub, and then create a pull request via your forked repository. More detail can be found [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

Note that a [code of conduct](https://github.com/handley-lab/anesthetic/blob/master/CODE_OF_CONDUCT.md) applies to all spaces managed by the anesthetic project, including issues and pull requests. 

## Contributing - `pre-commit`

anesthetic uses flake8 and pydocstyle to maintain a consistent style. To speed up linting, contributors can optionally use pre-commit to ensure their commits comply.

First, ensure that pre-commit, flake8 and pydocstyle are installed:
```
pip install pre-commit flake8 pydocstyle
```
Then install the pre-commit to the .git folder:
```
pre-commit install
```
This will check changed files, and abort the commit if they do not comply. To uninstall, run:
```
pre-commit uninstall
```
