from setuptools import setup

# Fork patch (Radiationsafety): static version >= upstream 1.5.1 so that
# "cuqipy>=1.5.0" requirements (e.g. bssunfold[cuqi]) resolve against the
# git-built distribution regardless of git tags / shallow clones.
setup(
    version="1.5.2",
)
