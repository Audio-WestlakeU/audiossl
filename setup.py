from setuptools import setup,find_packages


packages=find_packages(where=".")

setup(
    name='audiossl',
    version='0.1',
    description='',
    author='lixian',
    email='lixian@westlake.edu.cn',
    packages=packages,
    entry_points={
                  'console_scripts':[
                                    'atst_transfer_train=audiossl.methods.atst.transfer.train:main' 
                                    ]
                 }
      )