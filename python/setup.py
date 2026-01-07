import os
import setuptools


# This is to make sure that the package supports editable installs
version = "0.5.7"
maca_ai_version = os.environ.get('MACA_AI_VERSION', '0.0.0.0')
version += "+maca"+maca_ai_version
setuptools.setup(version=version)