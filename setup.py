# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import re

from setuptools import setup, find_packages

package_name = "autopk"
version = '0.1.0'

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme_markdown = f.read()
except:
    logging.warn("Failed to load %s" % readme_filename)
    readme_markdown = ""


if __name__ == '__main__':
    setup(
        name=package_name,
        version=version,
        py_modules=find_packages(),
        include_package_data=True,
        description="Automated model selection for pharmacokinetics",
        author="Sergey Feldman",
        author_email="sergey@data-cowboys.com",
        url='https://github.com/ki-tools/autopk',
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=[
            'click',
            'pandas',
            'numpy',
            'joblib',
            'cma',
            'statsmodels',
            'matplotlib',
            'seaborn'
        ],
        entry_points='''
            [console_scripts]
            autopk=autopk.scripts.autopk:cli
        ''',
        long_description=readme_markdown,
        long_description_content_type='text/markdown',
    )