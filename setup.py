from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


setup(name='gloveals',
      version='0.0.1',
      description='Glove-ALS',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      keywords='GloVe-ALS',
      url='http://github.com/eldrin/glove-als',
      author='Jaehun Kim',
      author_email='j.h.kim@tudelft.nl',
      license='MIT',
      packages=find_packages('.'),
      install_requires=requirements(),
      extras_require={
          'dev': []
      },
      test_suite='tests',
      zip_safe=False)