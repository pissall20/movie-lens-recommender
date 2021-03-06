import os
import urllib
import zipfile


complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

datasets_path = "/home/hduser/PycharmProjects/recommender/datasets"
print datasets_path

complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')

small_f = urllib.urlretrieve(small_dataset_url, small_dataset_path)
complete_f = urllib.urlretrieve(complete_dataset_url, complete_dataset_path)

with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)

with zipfile.ZipFile(complete_dataset_path, "r") as z:
    z.extractall(datasets_path)




