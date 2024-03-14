import sentencepiece as spm
import torch, json

def train_sentencepiece(file: str = 'shakespeare.txt') -> None:
	spm.SentencePieceTrainer.train(
	    input=file,
	    model_prefix='shakespeare-sp', 
	    model_type='bpe', 
	    vocab_size=128,
	    character_coverage=1.0,
	   # user_defined_symbols=[],
	)

def partitioning(model_file: str = 'politic4k-sp.model', file: str = 'politic4k.txt') -> None:
	sp = spm.SentencePieceProcessor(model_file=model_file)
	pname = file.split('.')[0]
	psize = 2048
	with open(file) as fp:
		lines = fp.readlines()
		nol = len(lines)
		b = nol // psize
		r = nol - (b * psize)
		for each in range(b + (1 if r else 0)):
			batch_lines = torch.tensor([])
			for line in lines[each * psize:(each * psize) + psize]:
				encode = torch.tensor(sp.encode(line + '\n'))
				batch_lines = torch.cat((batch_lines, encode), dim=0)
			json.dump(batch_lines.tolist(), open(f"{pname}_p{each}.json", 'w'))



if __name__ == '__main__':
	train_sentencepiece()
	# partitioning()
