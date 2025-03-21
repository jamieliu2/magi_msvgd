from setuptools import setup 

setup( 
	name='magi_msvgd', 
	version='1.0', 
	description='Mitotic Stein variational gradient descent for manifold-constrained Gaussian process inference.', 
	author='Jamie Liu', 
	packages=['magi_msvgd'], 
	install_requires=[ 
		'numpy',
		'scipy',
		'scikit-learn',
		'tqdm'
	], 
) 