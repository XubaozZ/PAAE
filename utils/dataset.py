import json
from os.path import join
from typing import Union, Sequence
from pathlib import Path
from typing import Tuple
import PIL
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy import io
from torch.utils.data import Dataset
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
		target = sample.target - 1
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

class ALGAE(VisionDataset):

    base_folder = ''

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(ALGAE, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.train = train


        split = 'train' if self.train else 'test'
        self.split_folder = os.path.join(self.root, self.base_folder, split)
        if not os.path.exists(self.split_folder):
            raise RuntimeError(f"{self.split_folder} not found. Please check dataset path.")


        self.class_to_idx = self._find_classes(self.split_folder)
        self.samples = self._make_dataset(self.split_folder, self.class_to_idx)


        self.class_names = list(self.class_to_idx.keys())


        cla_dict = dict((val, key) for key, val in self.class_to_idx.items())
        with open('categories_public.json', 'w') as f:
            json.dump(cla_dict, f, indent=4)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        return {cls_name: i for i, cls_name in enumerate(classes)}

    def _make_dataset(self, dir, class_to_idx):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if os.path.isfile(path):
                        item = (path, class_index)
                        instances.append(item)
        return instances

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

class Cars(VisionDataset):
	file_list = {
		'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims'),
		'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
	}

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train


		loaded_mat = scipy.io.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
		loaded_mat = loaded_mat['annotations'][0]
		self.samples = []
		for item in loaded_mat:
			if self.train != bool(item[-1][0]):
				path = str(item[0][0])
				label = int(item[-2][0]) - 1
				self.samples.append((path, label))

	def __getitem__(self, index):
		path, target = self.samples[index]
		path = os.path.join(self.root, path)

		image = self.loader(path)
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def __len__(self):
		return len(self.samples)

	def _check_exists(self):
		print(os.path.join(self.root, self.file_list['imgs'][1]))
		return os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))

	def _download(self):
		print('Downloading...')
		for url, filename in self.file_list.values():
			download_url(url, root=self.root, filename=filename)
		print('Extracting...')
		archive = os.path.join(self.root, self.file_list['imgs'][1])
		extract_archive(archive)


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


class Aircraft(Dataset):
	img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

	def __init__(self, root, train=True, transform=None):
		self.train = train
		self.root = root
		self.class_type = 'variant'
		self.split = 'trainval' if self.train else 'test'
		self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
		                                 'images_%s_%s.txt' % (self.class_type, self.split))
		self.transform = transform

		(image_ids, targets, classes, class_to_idx) = self.find_classes()
		samples = self.make_dataset(image_ids, targets)

		self.loader = default_loader

		self.samples = samples
		self.classes = classes
		self.class_to_idx = class_to_idx

	def __getitem__(self, index):
		path, target = self.samples[index]
		sample = self.loader(path)
		sample = self.transform(sample)
		return sample, target

	def __len__(self):
		return len(self.samples)

	def find_classes(self):

		image_ids = []
		targets = []
		with open(self.classes_file, 'r') as f:
			for line in f:
				split_line = line.split(' ')
				image_ids.append(split_line[0])
				targets.append(' '.join(split_line[1:]))


		classes = np.unique(targets)
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		targets = [class_to_idx[c] for c in targets]

		return image_ids, targets, classes, class_to_idx

	def make_dataset(self, image_ids, targets):
		assert (len(image_ids) == len(targets))
		images = []
		for i in range(len(image_ids)):
			item = (os.path.join(self.root, self.img_folder,
			                     '%s.jpg' % image_ids[i]), targets[i])
			images.append(item)
		return images


class NABirds(VisionDataset):
	base_folder = 'images'
	filename = 'nabirds.tar.gz'
	md5 = 'df21a9e4db349a14e2b08adfd45873bd'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
		super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
		if download is True:
			msg = ("The dataset is no longer publicly accessible. You need to "
			       "download the archives externally and place them in the root "
			       "directory.")
			raise RuntimeError(msg)
		elif download is False:
			msg = ("The use of the download flag is deprecated, since the dataset "
			       "is no longer publicly accessible.")
			warnings.warn(msg, RuntimeWarning)

		dataset_path = root


		self.loader = default_loader
		self.train = train

		image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
		                          sep=' ', names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
		                                 sep=' ', names=['img_id', 'target'])

		self.label_map = self.get_continuous_class_map(image_class_labels['target'])
		train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
		                               sep=' ', names=['img_id', 'is_training_img'])
		data = image_paths.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]


		self.class_names = self.load_class_names(dataset_path)
		self.class_hierarchy = self.load_hierarchy(dataset_path)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = self.label_map[sample.target]
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def get_continuous_class_map(self, class_labels):
		label_set = set(class_labels)
		return {k: i for i, k in enumerate(label_set)}

	def load_class_names(self, dataset_path=''):
		names = {}

		with open(os.path.join(dataset_path, 'classes.txt')) as f:
			for line in f:
				pieces = line.strip().split()
				class_id = pieces[0]
				names[class_id] = ' '.join(pieces[1:])

		return names

	def load_hierarchy(self, dataset_path=''):
		parents = {}

		with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
			for line in f:
				pieces = line.strip().split()
				child_id, parent_id = pieces
				parents[child_id] = parent_id

		return


class OxfordFlowers(Dataset):
	def __init__(self, root, train=True, transform=None):
		self.transform = transform
		self.root = root
		self.loader = default_loader
		train_set = pd.read_csv(os.path.join(self.root, 'train.txt'),
		                        sep=' ', names=['img_path', 'target'])
		test_set = pd.read_csv(os.path.join(self.root, 'test.txt'),
		                       sep=' ', names=['img_path', 'target'])
		if train:
			self.data = train_set
		else:
			self.data = test_set

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, sample.img_path)
		target = sample.target
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		return img, target

	def __len__(self):
		return len(self.data)


class OxfordIIITPet(VisionDataset):

	_RESOURCES = (
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
	)
	_VALID_TARGET_TYPES = ("category", "segmentation")

	def __init__(
			self,
			root: str,
			train: bool = True,
			transform: Optional[Callable] = None,
			target_types: Union[Sequence[str], str] = "category",
			target_transform: Optional[Callable] = None,
			download: bool = False,
	):
		if train:
			split = "trainval"
		else:
			split = "test"
		self._split = verify_str_arg(split, "split", ("trainval", "test"))
		if isinstance(target_types, str):
			target_types = [target_types]
		self._target_types = [
			verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
		]

		super().__init__(root, transform=transform, target_transform=target_transform)
		self._images_folder = os.path.join(self.root, "images")
		self._anns_folder = os.path.join(self.root, "annotations")
		self._segs_folder = os.path.join(self.root, "trimaps")

		if download:
			self._download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		image_ids = []
		self._labels = []
		with open(os.path.join(self._anns_folder, f"{self._split}.txt")) as file:
			for line in file:
				image_id, label, *_ = line.strip().split()
				image_ids.append(image_id)
				self._labels.append(int(label) - 1)

		self.classes = [
			" ".join(part.title() for part in raw_cls.split("_"))
			for raw_cls, _ in sorted(
				{(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
				key=lambda image_id_and_label: image_id_and_label[1],
			)
		]
		self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

		self._images = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
		self._segs = [os.path.join(self._segs_folder, f"{image_id}.png") for image_id in image_ids]

	def __len__(self) -> int:
		return len(self._images)

	def __getitem__(self, idx: int) -> Tuple[Any, Any]:
		image = Image.open(self._images[idx]).convert("RGB")

		target: Any = []
		for target_type in self._target_types:
			if target_type == "category":
				target.append(self._labels[idx])
			else:
				target.append(Image.open(self._segs[idx]))

		if not target:
			target = None
		elif len(target) == 1:
			target = target[0]
		else:
			target = tuple(target)

		if self.transform:
			image = self.transform(image)

		return image, target

	def _check_exists(self) -> bool:
		for folder in (self._images_folder, self._anns_folder):
			if not (os.path.exists(folder) and os.path.isdir(folder)):
				return False
		else:
			return True

	def _download(self) -> None:
		if self._check_exists():
			return

		for url, md5 in self._RESOURCES:
			download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


class Food101(VisionDataset):

	_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
	_MD5 = "85eeb15f3717b99a5da872d97d918f87"

	def __init__(
			self,
			root: str,
			train: bool = True,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			download: bool = False,
	) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform)
		split = "train" if train else "test"
		self._split = verify_str_arg(split, "split", ("train", "test"))
		self._base_folder = Path(self.root) / "food-101"
		self._meta_folder = self._base_folder / "meta"
		self._images_folder = self._base_folder / "images"

		if download:
			self._download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		self._labels = []
		self._image_files = []
		with open(self._meta_folder / f"{split}.json") as f:
			metadata = json.loads(f.read())

		self.classes = sorted(metadata.keys())
		self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

		for class_label, im_rel_paths in metadata.items():
			self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
			self._image_files += [
				self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
			]

	def __len__(self) -> int:
		return len(self._image_files)

	def __getitem__(self, idx) -> Tuple[Any, Any]:
		image_file, label = self._image_files[idx], self._labels[idx]
		image = PIL.Image.open(image_file).convert("RGB")

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label

	def extra_repr(self) -> str:
		return f"split={self._split}"

	def _check_exists(self) -> bool:
		return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

	def _download(self) -> None:
		if self._check_exists():
			return
		download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


if __name__ == '__main__':
	root = "./data"
	train_set = Food101(root, train=True, transform=None)
	print(len(train_set))
