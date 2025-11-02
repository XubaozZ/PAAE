from os.path import join
import pandas as pd
import scipy
from scipy import io
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import *


class CUB(VisionDataset):
	base_folder = 'CUB_200_2011/images'
	file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(CUB, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train
		if download:
			self._download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
		                     names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
		                                 sep=' ', names=['img_id', 'target'])
		train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
		                               sep=' ', names=['img_id', 'is_training_img'])

		data = images.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
		                          sep=' ', names=['class_name'], usecols=[1])
		self.class_names = class_names['class_name'].to_list()
		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def _check_integrity(self):
		try:
			self._load_metadata()
		except Exception:
			return False

		for index, row in self.data.iterrows():
			filepath = os.path.join(self.root, self.base_folder, row.filepath)
			if not os.path.isfile(filepath):
				print(filepath)
				return False
		return True

	def _download(self):
		import tarfile

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

		with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
			tar.extractall(path=self.root)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = sample.target - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target





class Dogs(VisionDataset):
	download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train

		if download:
			self.download()

		split = self.load_split()

		self.images_folder = join(self.root, 'Images')
		self.annotations_folder = join(self.root, 'Annotation')
		self._breeds = list_dir(self.images_folder)

		self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

		self._flat_breed_images = self._breed_images

	def __len__(self):
		return len(self._flat_breed_images)

	def __getitem__(self, index):
		image_name, target = self._flat_breed_images[index]
		image_path = join(self.images_folder, image_name)
		image = self.loader(image_path)

		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def download(self):
		import tarfile

		if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
			if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
				print('Files already downloaded and verified')
				return

		for filename in ['images', 'annotation', 'lists']:
			tar_filename = filename + '.tar'
			url = self.download_url_prefix + '/' + tar_filename
			download_url(url, self.root, tar_filename, None)
			print('Extracting downloaded file: ' + join(self.root, tar_filename))
			with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
				tar_file.extractall(self.root)
			os.remove(join(self.root, tar_filename))

	def load_split(self):
		if self.train:
			split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
		else:
			split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

		split = [item[0][0] for item in split]
		labels = [item[0] - 1 for item in labels]
		return list(zip(split, labels))

	def stats(self):
		counts = {}
		for index in range(len(self._flat_breed_images)):
			image_name, target_class = self._flat_breed_images[index]
			if target_class not in counts.keys():
				counts[target_class] = 1
			else:
				counts[target_class] += 1

		print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
		                                                             float(len(self._flat_breed_images)) / float(
			                                                             len(counts.keys()))))

		return counts



if __name__ == '__main__':
	root = r""
	train_set = CUB(root, train=True, transform=None)
	print(len(train_set))
