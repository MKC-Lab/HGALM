import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle


class SelectionDataset(Dataset):
	def __init__(self, file_path, context_transform, response_transform, concat_transform, sample_cnt=None, mode='poly'):
		print(f"[DEBUG] Loading dataset from {file_path}, sample_cnt={sample_cnt}, mode={mode}")
		self.context_transform = context_transform
		self.response_transform = response_transform
		self.concat_transform = concat_transform
		self.data_source = []
		self.mode = mode
		weak_responses = []
		wweak_responses = []
		neg_responses = []
		with open(file_path, encoding='utf-8') as f:
			group = {
				'context': None,
				'responses': [],
				'labels': []
			}
			for line in f:
				split = line.strip().split('\t')
				lbl, context, response = int(split[0]), split[1],split[2]
				if lbl == 1 and len(group['responses']) > 0:
					self.data_source.append(group)
					group = {
						'context': None,
						'responses': [],
						'labels': []

					}
					if sample_cnt is not None and len(self.data_source) >= sample_cnt:
						break
				elif lbl== -1:
					weak_responses.append(response)
				elif lbl== -2:
					wweak_responses.append(response)
				else:
					neg_responses.append(response)
				group['responses'].append(response)
				group['labels'].append(lbl)
				group['context'] = context
			if len(group['responses']) > 0:
				self.data_source.append(group)
		print(f"[DEBUG] Loaded {len(self.data_source)} samples")

	def __len__(self):
		return len(self.data_source)

	def __getitem__(self, index):
		group = self.data_source[index]
		context, responses, labels = group['context'], group['responses'], group['labels']
		if self.mode == 'cross' or self.mode == 'kdcross':
			transformed_text = self.concat_transform(context, responses)
			ret = transformed_text, labels
		else:
			transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
			transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
			ret = transformed_context, transformed_responses, labels

		return ret

	def batchify_join_str(self, batch):#collate_fn：用于批次数据的处理函数。它指定了如何将单个样本组合成一个批次。在你的代码中，使用了 train_dataset.batchify_join_str 函数作为 collate_fn，它可能是你自定义的一个函数。
		if self.mode == 'cross' or self.mode == 'kdcross':
			text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
			labels_batch = []
			for sample in batch:
				text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

				text_token_ids_list_batch.append(text_token_ids_list)
				text_input_masks_list_batch.append(text_input_masks_list)
				text_segment_ids_list_batch.append(text_segment_ids_list)

				labels_batch.append(sample[1])

			long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

			text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
				torch.tensor(t, dtype=torch.long) for t in long_tensors)

			labels_batch = torch.tensor(labels_batch, dtype=torch.long)
			return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch

		else:
			contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
			responses_token_ids_list_batch, responses_input_masks_list_batch = [], [], [],[]
			labels_batch = []
			for sample in batch:
				(contexts_token_ids_list, contexts_input_masks_list), (responses_token_ids_list, responses_input_masks_list) = sample[:2]

				contexts_token_ids_list_batch.append(contexts_token_ids_list)
				contexts_input_masks_list_batch.append(contexts_input_masks_list)

				responses_token_ids_list_batch.append(responses_token_ids_list)
				responses_input_masks_list_batch.append(responses_input_masks_list)

				labels_batch.append(sample[-1])

			long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch,
											responses_token_ids_list_batch, responses_input_masks_list_batch]

			contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
			responses_token_ids_list_batch, responses_input_masks_list_batch = (
				torch.tensor(t, dtype=torch.long) for t in long_tensors)
			labels_batch = torch.tensor(labels_batch, dtype=torch.long)
			return contexts_token_ids_list_batch, contexts_input_masks_list_batch,\
						  responses_token_ids_list_batch, responses_input_masks_list_batch,labels_batch
