from setuptools import setup

setup(
    name='gpparser',
    version='0.0.1',
    packages=['gpparser'],
    entry_points={
        'console_scripts': [
            'gpparser=gpparser.__main__:main'
        ]
    }
)
