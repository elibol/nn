from setuptools import setup


setup(name='nn',
      version='0.1',
      author='Melih Elibol',
      author_email="elibol@gmail.com",
      description=("Simple Neural Networks using Autograd."),
      packages=["nn"],
      install_requires=['scipy', 'numpy', 'matplotlib', 'nose', 'scikit-learn>=0.17', 'pandas', 'joblib', 'autograd'],
      )
