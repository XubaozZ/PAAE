from settings.defaults import _C
from settings.setup_functions import *
root = os.path.dirname(os.path.abspath(__file__))
config = _C.clone()
cfg_file = os.path.join('configs', 'swin-cub.yaml')

config = SetupConfig(config, cfg_file)
config.defrost()
config.write = True
config.train.checkpoint = True
config.misc.exp_name = f'{config.data.dataset}'
config.misc.log_name = f'Ours'
try:
	config.cuda_visible = '4,3,1,6,2,0' if int(os.environ['WORLD_SIZE']) > 2 else '0,1'
except:
	config.cuda_visible = '0,1'

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



