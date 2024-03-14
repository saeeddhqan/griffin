'''
Contains main methods for training a model.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from torch import Tensor, nn
import model
from torch.utils.tensorboard import SummaryWriter
import wandb, argparse, time, random, math, numpy, json
from contextlib import nullcontext
from typing import Union, Optional, Iterable, Any, NoReturn, ClassVar

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)

block_size = 256
dim = 64
params = {
	'block_size': block_size,
	'lr': 1e-3, # Learning rate
	'min_lr': 1e-4, # Min learning rate
	'beta1': 0.9,
	'beta2': 0.999, # The less, the more stable
	'decay_lr': False,
	'eval_step': 250, # Every n step, we do an evaluation.
	'iterations': 501, # Like epochs
	'eval_iterations': 200, # Do n step(s), and calculate loss.
	'batch_size': 32,
	'nlayers': 2,
	'accumulation_steps': 2,
	'dropout': 0.1,
	'dim': dim,
	'weight_decay': 0.001,
	'grad_clip': 1.0,
	'vocab_size': 0,
	'device': 'cuda' if torch.cuda.is_available() else 'cpu',
	'variation': '', # When we change something, change this to distinguish different variations.
	'workdir': 'workdir',
	'data_file': 'data/shakespeare.txt',
	'load': '',
	'action': 'train',
	'mode': 'train',
	'data_load': None,
	'wandb': False,
	'tensorboard': False,
	'save_checkpoint': False,
	'parameters': None,
	'details': '',
	'compile': False,
	'dtype': 'float16',
	'autocast': None,
	'bias': False,
	'init_weight': 'normal_',
	'topk': -1,
	'token_type': 'token',
	'health': 2, # 0 for nothing, 1 for vector values, 2 for weight values of all layers
	'layers_health': [],
}


def after_conf_init():
	'''
		boring
	'''
	if config.device == 'cuda':
		config.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else config.dtype
		# config.dtype = 'float32'
	ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
	config.autocast = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
	config.topk = None if config.topk <= 0 else config.topk

	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
	mode.conf_model.dim = config.dim
	mode.conf_model.smqa_q_heads = config.dim // 128
	mode.conf_model.smqa_head_dim = 128

class Config:
	def __init__(self, data_dict: dict) -> NoReturn:
		'''
			Given a data_dict, the class treats each key/val as an object.
			Parameters
			----------
			data_dict: dict
				a dict that key is a property and value is its value
		'''
		self.__data_dict__ = data_dict

	def __getattr__(self, k: Union[int, str, bytes]) -> Any:
		'''
			Given a key, it returns its data if it exists, otherwise None.
			Parameters
			----------
			k: str
				key
			Returns
			-------
			v: Union[any type]
				the value of the k
		'''
		if k in self.__data_dict__:
			return self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def __setattr__(self, k: Union[int, str, bytes], v: Any) -> NoReturn:
		if k == '__data_dict__':
			super().__setattr__(k, v)
		else:
			self.__data_dict__[k] = v


	def __delattr__(self, k: Union[int, str, bytes]) -> NoReturn:
		'''
			Given a key, it deletes it from data dict if it exists.
			Parameters
			----------
			k: str
				key that needs to be removed
		'''
		if k in self.__data_dict__:
			del self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def set_args(self, args: argparse.Namespace) -> NoReturn:
		'''
			Given an object of argparse, the method adds all the KVs to the data.
			Parameters
			----------
			args: argparse.Namespace
				parsed args object
		'''
		for kv in args._get_kwargs():
			k, v = kv
			if k in dir(model.conf_model):
				model.conf_model.k = v
			self.__setattr__(k, v)

		after_conf_init()


	def get_model_params(self, abstract: bool = False) -> dict:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			abstract: bool
				True if you want to remove metadata from dictionary.
		'''
		if abstract:
			filters = (
				'data_load', 'action', 'load', 'workdir',
				'wandb', 'tensorboard', 'details', 'data_file',
				'variation', 'device', 'mode', 'autocast',
				'flash_attention', 'compile',
				'init_weight', 'freqs_cis_test',
			)
		else:
			filters = ('data_load', 'load', 'iterations', 'autocast', 'freqs_cis_test')
		params = {}
		for k in self.__data_dict__:
			if k not in filters:
				params[k] = self.__data_dict__[k]
		return params


	def set_model_params(self, params: dict) -> NoReturn:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			params: dict
				Key value parameters.
		'''

		filters = (
			'data_load', 'action', 'load', 'workdir', 'mode')
		for k in params:
			if k not in filters:
				self.__data_dict__[k] = params[k]


class ManageModel:
	def __init__(self, model: ClassVar = None) -> NoReturn:
		'''
			Parameters
			----------
			model: Union[ClassVar, None]
				model instance
		'''
		self.model = model
		self.optimizer = None
		self.loss = {}
		self.best_loss = 1e9
		self.elapsed_time = 0
		self.scaler = None


	def get_lr(self, epoch, warmup_iters=2000, lr_decay_iters=3250):

		if epoch < warmup_iters:
			return config.lr # no warmup
			# return lr * epoch / warmup_iters

		if epoch > lr_decay_iters:
			return config.min_lr

		decay_ratio = (epoch - warmup_iters) / (lr_decay_iters - warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
		return config.min_lr + coeff * (config.lr - config.min_lr)


	def load_model(self, path: str) -> NoReturn:
		'''
			Load a model from path
			Parameters
			----------
			path: str
				Path to the model


		'''
		if not os.path.exists(path):
			print(f"Path '{path}' does not exist.")
			exit()
		checkpoint = torch.load(path)
		config.set_model_params(checkpoint['config'])
		config.data_load = model.Data(config)
		config.vocab_size = len(config.data_load)
		model.config = config
		self.model = model.Transformer()
		self.model.load_state_dict(checkpoint['model'])


	def net_health(self, epoch: int, lr: float, test_time: bool) -> NoReturn:
		'''
			Logging more information about the vectors, weights, 
			and one day gradients. Needs to be run after each iter.
			Parameters
			----------
			epoch: int
				current epoch
			lr: float
				current learning rate
			test_time: bool
				true if it's the test time, false otherwise
		'''
		for i, layer in enumerate(config.layers_health):
			for k, v in layer.items():
				if config.tensorboard:
					self.tensorboard_writer.add_scalar(f"layers.{i}.{k}", v, epoch, new_style=True)
				if config.wandb:
					wandb.log({
						f"layers.{i}.{k}": v,
					})


		if config.tensorboard and config.decay_lr:
			self.tensorboard_writer.add_scalar('lr', lr, epoch, new_style=True)
		if config.wandb:
			wandb.log({
				'lr': lr,
			})
		if test_time:
			for name, param in self.model.named_parameters():
				grad_norm = None if param.grad is None else param.grad.data.norm(2).item()
				weight_norm = None if 'weight' not in name else param.norm(2).item()
				if config.tensorboard:
					if grad_norm is not None:
						self.tensorboard_writer.add_scalar(f"{name}.gradient.norm", grad_norm, epoch, new_style=True)
					if weight_norm is not None:
						self.tensorboard_writer.add_scalar(f"{name}.weight.norm", weight_norm, epoch, new_style=True)

				if config.wandb:
					if grad_norm is not None:
						wandb.log({
							f"{name}.gradient.norm": grad_norm,
						})
					if weight_norm is not None:
						wandb.log({
							f"{name}.gradient.norm": weight_norm,
						})



		config.layers_health = []

		if config.tensorboard:
			self.tensorboard_writer.flush()


	def pre_train(self) -> NoReturn:
		'''
			Prepare the language model for training.
			Init optimizer, tensorboard, wandb, dirs, model, etc.
		'''
		self.model.train()
		self.model.to(config.device)

		if self.optimizer is None:
			use_fused = config.device == 'cuda'

			self.optimizer = torch.optim.AdamW(
				self.model.parameters(),
				lr=config.lr,
				# amsgrad=True, # Found amsgrad better.
				betas=(config.beta1, config.beta2),
				weight_decay=config.weight_decay,
				fused=use_fused,
			)

		ver = f'{config.variation}_griffin'

		variation = f"{ver}_{config.nlayers}nl_\
		{config.dim}d_{config.dropout}\
		do_{config.block_size}bs_{config.lr}lr_{int(config.decay_lr)}\
		dlr".strip().replace('\t', '').replace(' ', '')

		if config.tensorboard:
			self.tensorboard_writer = SummaryWriter(
				comment='_' + variation,
				filename_suffix='',
			)
		if config.wandb:
			self.wandb_init = wandb.init(
				project='griffin',
				name=variation,
				config=config.get_model_params(),
			)
		self.path_format = os.path.join(
			config.workdir,
			f"model_{variation}",
		)

		if config.wandb:
			self.wandb_init.watch(self.model, log='all')

		os.makedirs(config.workdir, exist_ok=True)
		self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))


	def pre_test(self) -> NoReturn:
		'''
			Prepare the language model for testing.
		'''
		self.model.eval()
		self.model.to(config.device)


	def post_train(self) -> NoReturn:
		'''
			Tasks that relate to after training happen here.

		'''
		if config.tensorboard:
			hyparams = config.get_model_params(abstract=True)
			metrics = {}
			hyparams['test_loss'] = self.loss['test'].item()
			hyparams['train_loss'] = self.loss['train'].item()
			hyparams['elapsed_time'] = round(self.elapsed_time / 60, 4)
			hyparams['parameters'] = config.parameters
			for i in hyparams:
				self.tensorboard_writer.add_text(i, str(hyparams[i]))
			self.tensorboard_writer.flush()
			self.tensorboard_writer.close()
		if config.wandb:
			wandb.log({
				'meta/params': config.parameters,
				'meta/elapsed_time': round(self.elapsed_time / 60, 4)
			})


	def post_test(self) -> NoReturn:
		pass


	@torch.no_grad()
	def calculate_loss(self, length: int) -> dict[str, int]:
		'''
			We select eval_iterations chunks from both train and test data
			and save their losses. All in all, evaluating the perf
			of the model on train and test data. Learnt from nanoGPT
			Parameters
			----------

			Returns
			-------
			loss: dict
				testing process loss
		'''

		self.model.eval()

		out = {}
		for split in ('train', 'test'):
			# A tensor to capture the losses
			losses = torch.zeros(config.eval_iterations)
			for k in range(config.eval_iterations):
				X, y = config.data_load.get_batch(0, split, block_size=length)
				with config.autocast:
					_, loss = self.model(X, y)
				losses[k] = loss.item()
			out[split] = losses.mean()

		self.model.train()

		return out


	@torch.no_grad()
	def test(self, epoch: int) -> NoReturn:
		'''
			Generate a sequence, calculate loss, and log
			Parameters
			----------
			epoch: int
				current epoch
		'''
		state = config.mode
		config.mode = 'inference'
		seq, elapsed, elapsed_per_token = self.generator(epoch=epoch)

		print(seq)
		print('-' * 10)
		print(f"[{epoch}] > Elapsed: {elapsed}")
		print(f"[{epoch}] > Elapsed per character: {elapsed_per_token}")

		self.loss = self.calculate_loss(config.block_size)
		test_loss = round(self.loss['test'].item(), 5)
		train_loss = round(self.loss['train'].item(), 5)
		test_pp = round(torch.exp(self.loss['test']).item(), 5)
		train_pp = round(torch.exp(self.loss['train']).item(), 5)
		print(f"[{epoch}] > train: {train_loss}, {train_pp} PP, test: {test_loss}, {test_pp} PP")
		print('-' * 30)

		if config.tensorboard:
			self.tensorboard_writer.add_scalar(f'train_loss', train_loss, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'test_loss', test_loss, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'train_pp', train_pp, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'test_pp', test_pp, epoch, new_style=True)
			self.tensorboard_writer.flush()

		if config.wandb:
			wandb.log({
				f'train/loss': train_loss,
				f'test/loss': test_loss,
				f'train/perplexity': train_pp,
				f'test/perplexity': test_pp,
				'iter': epoch,
			})

		config.mode = state


	def train_procedure(self) -> NoReturn:
		'''
			Running one iteration.
			Parameters
			----------
			Returns
			-------
			bool:
				specifies whether the training should continue or not.
		'''
		epoch = 0
		X, Y = config.data_load.get_batch(epoch)
		while True:
			test_time = epoch % config.eval_step == config.eval_step - 1
			lr = self.get_lr(epoch + 1) if config.decay_lr else config.lr

			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

			config.layers_health = [{} for _ in range(config.nlayers)]
			start = time.time()
			for accum_step in range(config.accumulation_steps):
				with config.autocast:
					pred, loss = self.model(X, Y)
					loss = loss / config.accumulation_steps

				X, Y = config.data_load.get_batch(epoch)
				self.scaler.scale(loss).backward()


			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(
				self.model.parameters(),
				config.grad_clip,
			)

			self.scaler.step(self.optimizer)
			self.scaler.update()
			# self.net_health(epoch, lr, test_time)
			self.optimizer.zero_grad(set_to_none=True)

			stop = time.time()
			self.elapsed_time += stop - start

			# If it's the right time to test the model
			if test_time:
				self.test(epoch)
				if config.save_checkpoint or self.loss['test'] < self.best_loss:
					self.best_loss = self.loss['test']
					# torch.save({
					# 	'model': self.model.state_dict(),
					# 	'optimizer': self.optimizer.state_dict(),
					# 	'config': config.get_model_params(),
					# 	'train_loss': self.loss['train'],
					# 	'test_loss': self.loss['test'],
					# 	'epoch': epoch,
					# 	}, self.path_format + f"_{epoch}.pt")
			epoch += 1

			if epoch > config.iterations:
				break



	def train(self) -> NoReturn:
		'''
			Training process.
		'''

		self.pre_train()

		try:
			self.train_procedure()
		except KeyboardInterrupt:
			print(f"Keyboard interrupt.")

		self.post_train()


	@torch.no_grad()
	def generator(self, seq_len: int = 100, epoch: int = 0) -> tuple[str, float, float]:
		'''
			Generate a sequence with seq_len length and return it
			along with time elapsed.
			Parameters
			----------
			seq_len: int
				sequence length you want to create
			Returns
			-------
			decoded: str
				generated sequence
			took: float
				elapsed time to generate the sequence
			took_per_token: float
				elapsed time to generate each token
		'''
		self.pre_test()

		X, _ = config.data_load.get_batch(0, 'test', batch_size=1)

		start = time.time()

		with config.autocast:
			generated = self.model.autocomplete(X, seq_len, top_k=config.topk)
		end = time.time()
		decoded = config.data_load.decode(generated[0].tolist())
		took = end - start
		took_per_token = took / len(decoded)
		self.post_test()

		return decoded, took, took_per_token


if __name__ == '__main__':
	config = Config(params)
	parser = argparse.ArgumentParser()
	parser.add_argument('--action', '-a', type=str, help='train, and test', required=True)
	parser.add_argument('--device', type=str, default=config.device, help=f"device type, default {config.device}")
	parser.add_argument('--workdir', type=str, default=config.workdir, help=f"directory to save models, default {config.device}")
	parser.add_argument('--load', type=str, default=config.load, help='path to a model to start with')
	parser.add_argument('--data-file', type=str, default=config.data_file, help=f"input data file, default {config.data_file}")
	parser.add_argument('--variation', '-v', type=str, default=config.variation, help=f"model variation, default {config.variation}")
	parser.add_argument('--details', type=str, help=f"model details, default {config.details}")
	parser.add_argument('--iterations', '-i', type=int, default=config.iterations, help=f"number of training iterations, default {config.iterations}")
	parser.add_argument('--lr', '-lr', type=float, default=config.lr, help=f"learning rate, default {config.lr}")
	parser.add_argument('--min-lr', '-ml', type=float, default=config.min_lr, help=f"minimum learning rate, default {config.min_lr}")
	parser.add_argument('--dropout', '-do', type=float, default=config.dropout, help=f"dropout prob, default {config.dropout}")
	parser.add_argument('--nlayers', '-nl', type=int, default=config.nlayers, help=f"number of blocks, default {config.nlayers}")
	parser.add_argument('--dim', '-d', type=int, default=config.dim, help=f"embedding size, default {config.dim}")
	parser.add_argument('--accumulation-steps', '-as', type=int, default=config.accumulation_steps, help=f"accumulation steps, default {config.accumulation_steps}")
	parser.add_argument('--block-size', '-bs', type=int, default=config.block_size, help=f"length input sequence, default {config.block_size}")
	parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help=f"batch size, default {config.batch_size}")
	parser.add_argument('--topk', type=int, default=config.topk, help=f"topk sampling, default {config.topk}")
	parser.add_argument('--wandb', action='store_true', default=config.wandb, help=f"use wandb for visualization, default {config.wandb}")
	parser.add_argument('--tensorboard', action='store_true', default=config.tensorboard, help=f"use tensorboard for visualization, default {config.tensorboard}")
	parser.add_argument('--compile', action='store_true', default=config.compile, help=f"compile the model for faster training, default {config.compile}")
	parser.add_argument('--decay-lr', action='store_true', default=config.decay_lr, help=f"decay learning rate, default {config.decay_lr}")
	args = parser.parse_args()

	config.set_args(args)
	task = ManageModel()

	match config.action:
		case 'train':
			config.mode = 'train'
			if config.load != '':
				task.load_model(config.load)
			else:
				config.data_load = model.Data(config)
				model.config = config
				the_model = model.Griffin()
				task.model = torch.compile(the_model) if config.compile else the_model
			task.train()
		case 'test':
			config.mode = 'inference'
			task.load_model(config.load)
			seq, elapsed, elapsed_per_token = task.generator(500)
			print(seq)
			print('-' * 12)
			print('Elapsed:', elapsed)
			print('Elapsed per character:', elapsed_per_token)
		case _:
			print('Invalid action.')
