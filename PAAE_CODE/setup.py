from settings.defaults import _C
from settings.setup_functions import *
root = os.path.dirname(os.path.abspath(__file__))
config = _C.clone()


cfg_file = os.environ.get('PAAE_CONFIG', os.path.join('configs', 'swin-cub.yaml'))
config = SetupConfig(config, cfg_file)
config.defrost()


if os.environ.get('PAAE_DATA_ROOT'):
	config.data.data_root = os.environ['PAAE_DATA_ROOT']
if os.environ.get('PAAE_RESUME'):
	config.model.resume = os.environ['PAAE_RESUME']
if os.environ.get('PAAE_OUTPUT'):
	config.misc.output = os.environ['PAAE_OUTPUT']
if os.environ.get('PAAE_EVAL_MODE'):
	config.misc.eval_mode = os.environ['PAAE_EVAL_MODE'].lower() in ['1', 'true', 'yes']
if os.environ.get('PAAE_THROUGHPUT'):
	config.misc.throughput = os.environ['PAAE_THROUGHPUT'].lower() in ['1', 'true', 'yes']


config.write = True
config.train.checkpoint = True
config.misc.exp_name = f'{config.data.dataset}'
config.misc.log_name = os.environ.get('PAAE_LOG_NAME', 'Ours')


if os.environ.get('PAAE_CUDA_VISIBLE'):
	config.cuda_visible = os.environ['PAAE_CUDA_VISIBLE']
else:
	try:
		config.cuda_visible = '0,1' if int(os.environ['WORLD_SIZE']) > 1 else '0'
	except Exception:
		config.cuda_visible = '0'


config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = os.path.join(config.model.pretrained,
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'


config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)
config.train.lr = ScaleLr(config)
log = SetupLogs(config, config.local_rank)
if config.write and config.local_rank in [-1, 0]:
	with open(config.data.log_path + '/config.json', "w") as f:
		f.write(config.dump())
config.freeze()
SetSeed(config)
