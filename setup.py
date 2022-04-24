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
                                    'atst_train=audiossl.methods.atst.train:main' ,
                                    'atst_downstream_train_freeze=audiossl.methods.atst.downstream.train_freeze:main' ,
                                    'atst_downstream_train_finetune=audiossl.methods.atst.downstream.train_finetune:main' 
                                    ]
                 }
      )
