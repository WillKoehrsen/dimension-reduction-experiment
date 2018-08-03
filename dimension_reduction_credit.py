import pandas as pd
import numpy as np

import dill
from umap import UMAP

from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE


from timeit import default_timer as timer

data_dir_train = '../input/features_no_duplicates_2018-07-26/features_no_duplicates_2018-07-26_train.data'
data_dir_test = '../input/features_no_duplicates_2018-07-26/features_no_duplicates_2018-07-26_test.data'


with open(data_dir_train, 'rb') as f:
	train = dill.load(f)

with open(data_dir_test, 'rb') as f:
	test = dill.load(f)

train_ids = list(train.index)

# train.drop(columns = ['SK_ID_CURR'], inplace = True)
# test.drop(columns = ['SK_ID_CURR'], inplace = True)

labels = np.array(train.pop('TARGET'))
train, test = train.align(test, axis = 1, join = 'inner')
print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)

print('Training shape: ', train.shape, file=open('../dimension_reduction/dimension_reduction_times.txt', 'w'))
print('Testing shape: ', test.shape, file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

labels_df = pd.DataFrame({'SK_ID_CURR': train_ids, 'TARGET': labels})
labels_df.to_csv('../dimension_reduction/labels.csv', index = False)

train = train.replace({-np.inf:np.nan, np.inf: np.nan})
test = test.replace({-np.inf:np.nan, np.inf: np.nan})

imputer = Imputer(strategy = 'median')
train = imputer.fit_transform(train)
test = imputer.transform(test)

n_components = 3

umap_ = UMAP(n_components = n_components)
pca = PCA(n_components = n_components)
ica = FastICA(n_components = n_components)
tsne = TSNE(n_components = n_components)

for method, name in zip([pca, ica, umap_, umap_, tsne], ['pca', 'ica', 'umap', 'umap_labels', 'tsne']):

	if name == 'umap_labels':
		print('Starting {}'.format(name.upper()))
		print('Starting {}'.format(name.upper()), 
			file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

		train_start = timer()
		train_transform = method.fit_transform(train, labels)
		train_end = timer()

		print('Training: {}: {} seconds.'.format(name.upper(), round(train_end - train_start, 2)))
		print('Training: {}: {} seconds.'.format(name.upper(), round(train_end - train_start, 2)), 
			file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

	else:
		print('Starting {}'.format(name.upper()))
		print('Starting {}'.format(name.upper()), 
			file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

		train_start = timer()
		train_transform = method.fit_transform(train)
		train_end = timer()
		print('Training: {}: {} seconds.'.format(name.upper(), round(train_end - train_start, 2)))
		print('Training: {}: {} seconds.'.format(name.upper(), round(train_end - train_start, 2)), 
			file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

	train_transform_df = pd.DataFrame(train_transform, columns =['%s_1' % name, '%s_2' % name, '%s_3' % name])
	train_transform_df.to_csv('../dimension_reduction/%s_train.csv' % name, index = False)

	if name != 'tsne':
		test_start = timer()
		test_transform = method.transform(test)
		test_end = timer()
		print('Testing: {}: {} seconds.'.format(name.upper(), round(test_end - test_start, 2)))
		print('Testing: {}: {} seconds.'.format(name.upper(), round(test_end - test_start, 2)), 
			file=open('../dimension_reduction/dimension_reduction_times.txt', 'a'))

		test_transform_df = pd.DataFrame(test_transform, columns = ['%s_1' % name, '%s_2' % name, '%s_3' % name])
		test_transform_df.to_csv('../dimension_reduction/%s_test.csv' % name, index = False)
