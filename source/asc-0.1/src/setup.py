from distutils.core import setup

setup(
    name="ASC-modules",
    version="0.1",
    description="ASC package",
    author="ICSE",
    license="MIT",
    packages=["asc", "asc/core", "asc/models",
            "asc/fir_filters", "asc/utils", "asc/frequency_analysis"]
)
