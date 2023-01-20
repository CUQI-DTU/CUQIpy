from setuptools import setup
import versioneer

setup(
    name="versioneer", 
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
