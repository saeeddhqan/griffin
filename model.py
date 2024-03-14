
from einops import rearrange, repeat, einsum
from dataclasses import dataclass
import torch, math, sentencepiece, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple

from accelerated_scan.warp import scan
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


config = None


@dataclass
class ModelConfig:
	vocab_size: int = 256
	num_layers: int = 1
	dim: int = 512
	smqa_head_dim: int = 128
	smqa_q_heads: int = 4
	smqa_kv_heads: int = 1
	smqa_window_size: int = 512
	hawk_expansion_factor: float = 1.5
	hawk_kernel_size: int = 4
	gmlp_expansion_factor: float = 2



conf_model = ModelConfig()

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		if config.token_type == 'token':
			data = torch.tensor([])
			if False:
				sp_model = 'data/wikisplit/wikisplit-sp.model'
				self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=sp_model)
				self.encode = self.tokenizer.encode
				parts = 62
				for part in range(parts):
					with open(f'data/wikisplit/train_p{part}.json') as fp:
						load = torch.tensor(json.load(fp))
						data = torch.cat((data, load), dim=0)
				data = data.to(torch.long)

			else:
				sp_model = 'data/shakespeare-sp.model'
				self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=sp_model)
				self.encode = self.tokenizer.encode
				with open(f'data/shakespeare.txt') as fp:
					encoded = self.encode(fp.read())
				data = torch.tensor(encoded, dtype=torch.long)

			self.decode = lambda seq: self.tokenizer.decode(seq)
			self.vocab_size = self.tokenizer.vocab_size()
		else:
			with open(config.data_file) as fp:
				text = fp.read()
			self.chars = sorted(list(set(text)))
			self.vocab_size = len(self.chars)
			self.stoi = {c:i for i,c in enumerate(self.chars)}
			self.itos = {i:c for c,i in self.stoi.items()}
			self.encode = lambda s: [self.stoi[x] for x in s]
			self.decode = lambda e: ''.join([self.itos[x] for x in e])
			data = torch.tensor(self.encode(text), dtype=torch.long)

		config.vocab_size = self.vocab_size
		conf_model.vocab_size = config.vocab_size
		train_split = int(0.9 * len(data))
		self.train_data = data[:train_split]
		self.test_data = data[train_split:]
		self.block_size = config.block_size
		self.batch_size = config.batch_size


	def __len__(self) -> int:
		return self.vocab_size


	def get_batch(self,
		idx: int, split: str = 'train',
		block_size = None,
		batch_size: int = -1,
	) -> tuple[Tensor, Tensor]:
		block_size = self.block_size if block_size is None else block_size
		batch_size = self.batch_size if batch_size == -1 else batch_size

		data = self.train_data if split == 'train' else self.test_data
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([data[i:i + block_size] for i in ix])
		y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
		return (x.pin_memory().to(config.device, non_blocking=True),
				y.pin_memory().to(config.device, non_blocking=True),
		)


class RMSNorm(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.scale = dim**-0.5
		self.gamma = nn.Parameter(torch.ones(dim))

	def forward(self, x: Tensor) -> Tensor:
		x = x / x.norm(p=2, dim=-1, keepdim=True)
		return self.gamma / self.scale * x


class Hawk(nn.Module):
	def __init__(self, dim: int = 1024, expansion_factor: int = 1.5, kernel_size: int = 4):
		super().__init__()
		hidden = int(dim * expansion_factor)
		self.input = nn.Linear(dim, 2 * hidden, bias=False)
		self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
							  kernel_size=kernel_size, groups=hidden, padding=kernel_size - 1)
		self.gates = nn.Linear(hidden, 2 * hidden, bias=True)
		self.forget_base = nn.Parameter(torch.linspace(-4.323, -9, hidden))
		self.output = nn.Linear(hidden, dim, bias=False)

		with torch.no_grad():
			self.input.weight.normal_(std=dim ** -0.5)
			self.gates.weight.normal_(std=hidden ** -0.5)
			self.output.weight.normal_(std=hidden ** -0.5)

	def forward(self, x: Tensor) -> Tensor:
		_N, T, _C = x.shape
		gate, x = self.input(x).chunk(2, dim=-1)
		x = self.conv(x.mT)[..., :T].mT

		# RG-LRU: linear recurrent unit with input-dependent gating
		forget, _input = self.gates(x).chunk(2, dim=-1)
		alpha = (-8 * F.softplus(self.forget_base) * forget.sigmoid()).exp()
		beta = (1 - alpha ** 2 + 1e-6).sqrt()
		x = beta * _input.sigmoid() * x

		h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
		x = self.output(F.gelu(gate) * h)
		return x


class GatedMLP(nn.Module):
	def __init__(self, dim: int = 1024, expansion_factor: int = 2):
		super().__init__()
		hidden = int(dim * expansion_factor)
		self.grow = nn.Linear(dim, 2 * hidden, bias=False)
		self.shrink = nn.Linear(hidden, dim, bias=False)

		with torch.no_grad():
			self.grow.weight.normal_(std=dim ** -0.5)
			self.shrink.weight.normal_(std=hidden ** -0.5)

	def forward(self, x: Tensor) -> Tensor:
		gate, x = self.grow(x).chunk(2, dim=-1)
		x = F.gelu(gate) * x
		return self.shrink(x)


class SlidingMQA(nn.Module):
	def __init__(self, dim: int = 1024,
		head_dim: int = 128, q_heads: int = 8,
		kv_heads: int = 1, window_size: int = 1024):
		super().__init__()
		self.head_dim = head_dim
		self.window_size = window_size
		self.rotary = RotaryEmbedding(dim=head_dim)
		self.q = nn.Linear(dim, head_dim * q_heads, bias=False)
		self.kv = nn.Linear(dim, 2 * head_dim * kv_heads, bias=False)
		self.output = nn.Linear(dim, dim, bias=False)

		with torch.no_grad():
			self.q.weight.normal_(std=dim ** -0.5)
			self.kv.weight.normal_(std=dim ** -0.5)
			self.output.weight.normal_(std=dim ** -0.5)

	def forward(self, x: Tensor) -> Tensor:
		N, T, C = x.shape
		q = self.q(x).view(N, T, -1, self.head_dim)
		kv = self.kv(x).view(N, T, 2, -1, self.head_dim)
		q, kv = self.rotary(q, kv)
		x = flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], causal=True, window_size=(-self.window_size, 0))
		x = x.view(N, T, C)
		return self.output(x)


class GriffinBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.hawk_norm = RMSNorm(dim=conf_model.dim)
		self.hawk = Hawk(dim=conf_model.dim, expansion_factor=conf_model.hawk_expansion_factor, kernel_size=conf_model.hawk_kernel_size)
		self.hawk_gmlp_norm = RMSNorm(dim=conf_model.dim)
		self.hawk_gmlp = GatedMLP(dim=conf_model.dim, expansion_factor=conf_model.gmlp_expansion_factor)

		self.smqa_norm = RMSNorm(dim=conf_model.dim)
		self.smqa = SlidingMQA(dim=conf_model.dim, head_dim=conf_model.smqa_head_dim, q_heads=conf_model.smqa_q_heads,
							   kv_heads=conf_model.smqa_kv_heads, window_size=conf_model.smqa_window_size)
		self.smqa_gmlp_norm = RMSNorm(dim=conf_model.dim)
		self.smqa_gmlp = GatedMLP(dim=conf_model.dim, expansion_factor=conf_model.gmlp_expansion_factor)

	def forward(self, x: Tensor) -> Tensor:
		x = x + self.hawk(self.hawk_norm(x))
		x = x + self.hawk_gmlp(self.hawk_gmlp_norm(x))
		x = x + self.smqa(self.smqa_norm(x))
		x = x + self.smqa_gmlp(self.smqa_gmlp_norm(x))
		return x


class Griffin(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
			dropout=nn.Dropout(config.dropout),
			ln1=RMSNorm(self.dim),
			lm_head=nn.Linear(self.dim, config.vocab_size, bias=False),
		))
		self.blocks = nn.ModuleList([GriffinBlock() for idx in range(config.nlayers)])
		self.stack.tok_embs.weight = self.stack.lm_head.weight
		self.apply(self.norm_weights)
		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params
		print("Number of parameters: %.3fM" % (self.count_params,))


	def num_params(self) -> int:
		n_params = sum(p.numel() for p in self.parameters())
		n_params -= self.stack.tok_embs.weight.numel()
		return n_params


	def norm_weights(self, module):
		# if isinstance(module, nn.Linear):
		# 	if config.init_weight == 'normal_':
		# 		nn.init.normal_(module.weight, mean=0.0, std=0.02)
		# 	else:
		# 		nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
		# 	if module.bias is not None:
		# 		 nn.init.constant_(module.bias, 0.001)
		if isinstance(module, nn.Embedding):
			if config.init_weight == 'normal_':
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			else:
				nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
		elif isinstance(module, nn.LayerNorm):
			nn.init.zeros_(module.bias)
			nn.init.ones_(module.weight)


	def forward(self, 
		seq: Tensor,
		targets: Union[Tensor, None] = None,
	) -> tuple[Tensor, Tensor]:

		x = self.stack.dropout(self.stack.tok_embs(seq))

		for i, block in enumerate(self.blocks):
			x = block(x)

		if targets is None:
			x = x[:,-1]

		x = self.stack.lm_head(self.stack.ln1(x))

		if targets is None:
			loss = None
		else:
			x = x.view(-1, config.vocab_size)
			loss = F.cross_entropy(x, targets.flatten())

		return x, loss


	def autocomplete(self, 
		idx: Tensor,
		_len: int = 10,
		temperature: float = 1.0,
		top_k: int = None,
	) -> Tensor:
		config.mode = 'inference'
		for _ in range(_len):
			idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
			logits, _ = self(idx_cond)
			logits = logits / temperature
			probs = F.softmax(logits, dim=-1)
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = -float('Inf')
			next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
		return idx
