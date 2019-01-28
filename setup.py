from setuptools import setup

setup(
    name='gpparser',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'gpparser=gpparser:main'
        ]
    }
)
