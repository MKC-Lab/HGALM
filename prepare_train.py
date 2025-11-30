import json
import argparse
import os
from collections import defaultdict
import random
from tqdm import tqdm

# P->P and P<-P
def no_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(tqdm(fin)):
			data = json.loads(line)
			doc = data['paper']

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(tqdm(doc2meta)):
			# sample positive
			dps = [x for x in doc2meta[doc] if x in doc2text]
			if len(dps) == 0:
				continue
			dp = random.choice(dps)

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


# PAP, PVP, P->P<-P, and P<-P->P
def one_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(tqdm(fin)):
			data = json.loads(line)
			doc = data['paper']

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(tqdm(doc2meta)):
			# sample positive
			metas = doc2meta[doc]
			dps = []
			for meta in metas:
				candidates = list(meta2doc[meta])
				if len(candidates) > 1:
					while True:
						dp = random.choice(candidates)
						if dp != doc:
							dps.append(dp)
							break
			if len(dps) == 0:
				continue
			dp = random.choice(dps)

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	

			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


def one_new_intermediate_node(dataset, doc2text, docs, metadata):
	meta2doc = defaultdict(set)
	doc2meta = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(tqdm(fin)):
			data = json.loads(line)
			doc = data['paper']

			metas = data[metadata]
			if not isinstance(metas, list):
				metas = [metas]
			for meta in metas:
				meta2doc[meta].add(doc)
			doc2meta[doc] = set(metas)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(tqdm(doc2meta)):
			metas = doc2meta[doc]  #doc的参考文献列表
			# Sample strong positive
			strong_candidates = []
			for meta in metas:
				candidates = list(meta2doc[meta]) #参考meta的doc
				if len(candidates) > 1:
					for dp in candidates:
						if dp != doc:
							strong_candidates.append(dp) 
							break
			if len(strong_candidates)==0:
				continue
			sp_doc = random.choice(strong_candidates) #sp_doc 参考 meta  doc 参考meta

			# Sample weak positive
			weak_candidates = []
			for meta in doc2meta[sp_doc]: #meta1 是 spdoc的参考文献
				candidates = list(meta2doc[meta]) #参考meta1的doc
				if(len(candidates)>1):
					for dp in candidates:
						if dp != doc and dp != sp_doc:
							weak_candidates.append(dp)
							break

			if len(weak_candidates) == 0:
				# 还没有，直接pass
				continue
			wp_doc = random.choice(weak_candidates) # wp_doc 参考meta1 sp_doc也参考meta1
			# # Sample secondary weak positive
			secondary_weak_candidates = []
			for meta in doc2meta[wp_doc]:
				candidates = list(meta2doc[meta])
				if(len(candidates)>1):
					for dp in candidates:
						if dp != doc and dp!= sp_doc and dp!=wp_doc:
							secondary_weak_candidates.append(dp)
							break
			if len(secondary_weak_candidates) == 0:
				# 没有pass
				continue
			swp_doc = random.choice(secondary_weak_candidates)
			# Sample negative
			while True:
				dn = random.choice(docs)
				# if dn != doc and dn !=sp_doc and dn!=wp_doc:
				if dn != doc and dn !=sp_doc and dn!=wp_doc and dn!=swp_doc:
					break

			# Write to file
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[sp_doc]}\n')
			fout.write(f'-1\t{doc2text[doc]}\t{doc2text[wp_doc]}\n')
			fout.write(f'-2\t{doc2text[doc]}\t{doc2text[swp_doc]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


# P(AA)P, P(AV)P, P->(PP)<-P, and P<-(PP)->P
def two_intermediate_node(dataset, doc2text, docs, metadata1, metadata2):
	meta12doc = defaultdict(set)
	doc2meta1 = {}
	doc2meta2 = {}
	with open(f'{dataset}/{dataset}_train.json') as fin:
		for idx, line in enumerate(tqdm(fin)):
			data = json.loads(line)
			doc = data['paper']

			meta1s = data[metadata1]
			if not isinstance(meta1s, list):
				meta1s = [meta1s]
			for meta1 in meta1s:
				meta12doc[meta1].add(doc)
			doc2meta1[doc] = set(meta1s)

			meta2s = data[metadata2]
			if not isinstance(meta2s, list):
				meta2s = [meta2s]
			doc2meta2[doc] = set(meta2s)

	with open(f'{dataset}_input/dataset.txt', 'w') as fout:
		for idx, doc in enumerate(tqdm(doc2meta1)):
			# sample positive
			meta1s = doc2meta1[doc]
			dps = []
			for meta1 in meta1s:
				candidates = []
				for d_cand in list(meta12doc[meta1]):
					if d_cand == doc:
						continue
					meta_intersec = doc2meta2[doc].intersection(doc2meta2[d_cand])
					if metadata1 != metadata2:
						if len(meta_intersec) >= 1:
							candidates.append(d_cand)
					else:
						if len(meta_intersec) >= 2:
							candidates.append(d_cand)
				if len(candidates) > 1:
					while True:
						dp = random.choice(candidates)
						if dp != doc:
							dps.append(dp)
							break
			if len(dps) == 0:
				continue
			dp = random.choice(dps)

			# sample negative
			while True:
				dn = random.choice(docs)
				if dn != doc and dn != dp:
					break	
					
			fout.write(f'1\t{doc2text[doc]}\t{doc2text[dp]}\n')
			fout.write(f'0\t{doc2text[doc]}\t{doc2text[dn]}\n')


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', type=str)
parser.add_argument('--metagraph', default='PRP', type=str)
args = parser.parse_args()

dataset = args.dataset
metagraph = args.metagraph

doc2text = {}
docs = []
with open(f'{dataset}/{dataset}_train.json') as fin:
	for idx, line in enumerate(tqdm(fin)):
		data = json.loads(line)
		doc = data['paper']
		text = data['text'].replace('_', ' ')
		doc2text[doc] = text
		docs.append(doc)

# P->P
if metagraph == 'PR':
	no_intermediate_node(dataset, doc2text, docs, 'reference')
# P<-P
elif metagraph == 'PC':
	no_intermediate_node(dataset, doc2text, docs, 'citation')
# PAP
elif metagraph == 'PAP':
	one_new_intermediate_node(dataset, doc2text, docs, 'author')
# PVP
elif metagraph == 'PVP':
	one_intermediate_node(dataset, doc2text, docs, 'venue')
# P->P<-P
elif metagraph == 'PRP':
	one_intermediate_node(dataset, doc2text, docs, 'reference')
# P<-P->P
elif metagraph == 'PCP':
	one_intermediate_node(dataset, doc2text, docs, 'citation')
# P(AA)P
elif metagraph == 'PAAP':
	two_intermediate_node(dataset, doc2text, docs, 'author', 'author')
# P(AV)P
elif metagraph == 'PAVP':
	two_intermediate_node(dataset, doc2text, docs, 'author', 'venue')
# P->(PP)<-P
elif metagraph == 'PRRP':
	two_intermediate_node(dataset, doc2text, docs, 'reference', 'reference')
# P<-(PP)->P
elif metagraph == 'PCCP':
	two_intermediate_node(dataset, doc2text, docs, 'citation', 'citation')
else:
	print('Wrong Meta-path/Meta-graph!!')