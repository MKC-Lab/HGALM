import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast,AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from adapters import AutoAdapterModel
from dataset import SelectionDataset
from transform import SelectionSequentialTransform, SelectionJoinTransform, SelectionConcatTransform
from encoder_kd import BiEncoder, CrossEncoder, kdCrossEncoder
import copy
import collections
from sklearn.metrics import label_ranking_average_precision_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


import logging
logging.basicConfig(level=logging.ERROR)

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

def eval_running_model(dataloader, test=False):
	model.eval()
	eval_loss, eval_hit_times = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	r1 = 0
	for step, batch in enumerate(dataloader):
		batch = tuple(t.to(device) for t in batch)
		if args.architecture == 'cross' or args.architecture =='kdcross':
			text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch = batch
			with torch.no_grad():
				logits = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch)
				loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))
		else:
			context_token_ids_list_batch, context_input_masks_list_batch, response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
			with torch.no_grad():
				logits = model(context_token_ids_list_batch, context_input_masks_list_batch,
							 response_token_ids_list_batch, response_input_masks_list_batch)
				loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))

		r1 += (logits.argmax(-1) == 0).sum().item()
		eval_loss += loss.item()
		nb_eval_examples += labels_batch.size(0)
		nb_eval_steps += 1
	
	eval_loss = eval_loss / nb_eval_steps
	eval_accuracy = r1 / nb_eval_examples
	
	if not test:
		result = {
			'train_loss': tr_loss / nb_tr_steps,
			'eval_loss': eval_loss,
			'R1': r1 / nb_eval_examples,
			'epoch': epoch,
			'global_step': global_step,
		}
	else:
		result = {
			'eval_loss': eval_loss,
			'R1': r1 / nb_eval_examples,
		}

	return result

def pred_running_model(dataloader, out_file):
	model.eval()
	with open(out_file, 'w') as fout:
		for step, batch in enumerate(dataloader):
			batch = tuple(t.to(device) for t in batch)
			if args.architecture == 'cross' or args.architecture =='kdcross':
				text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch = batch
				with torch.no_grad():
					logits = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch)
					# loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))
					for x in logits.flatten().cpu():
						fout.write(str(x.item())+'\n')
			else:
				context_token_ids_list_batch, context_input_masks_list_batch, response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
				with torch.no_grad():
					logits = model(context_token_ids_list_batch, context_input_masks_list_batch,
							 response_token_ids_list_batch, response_input_masks_list_batch)
					for x in logits.flatten().cpu():
						fout.write(str(x.item())+'\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	## Required parameters
	parser.add_argument("--bert_model", required=True, type=str)
	parser.add_argument("--eval", action="store_true")
	parser.add_argument("--model_type", default='bert', type=str)
	parser.add_argument("--output_dir", required=True, type=str)
	parser.add_argument("--train_dir", required=True, type=str)
	parser.add_argument("--test_file", required=True, type=str)

	parser.add_argument("--use_pretrain", action="store_true")
	parser.add_argument("--architecture", required=True, type=str, help='[bi, cross,kdcross]')

	parser.add_argument("--max_contexts_length", default=256, type=int)
	parser.add_argument("--max_response_length", default=256, type=int)
	parser.add_argument("--train_batch_size", default=0, type=int, help="Total batch size for training.")
	parser.add_argument("--eval_batch_size", default=0, type=int, help="Total batch size fo"
																	   "r eval.")
	parser.add_argument("--print_freq", default=500, type=int, help="Log frequency")

	parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.01, type=float)
	parser.add_argument("--warmup_steps", default=100, type=float)
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

	parser.add_argument("--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")
	parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--poly_m", default=0, type=int)
	parser.add_argument("--cuda", default=0, type=str)
	parser.add_argument("--kd_freq", default=2000, type=int)
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
				  "See details at https://nvidia.github.io/apex/amp.html",
	)
	args = parser.parse_args()
	print(args)
	set_seed(args)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
	if args.train_batch_size == 0:
		if args.architecture == 'bi':
			args.train_batch_size = 8
		else:
			args.train_batch_size = 4
	
	if args.eval_batch_size == 0:
		if args.architecture == 'bi':
			args.eval_batch_size = 256
		else:
			args.eval_batch_size = 128

	MODEL_CLASSES = {
		'bert': (BertConfig, BertTokenizerFast, BertModel),
	}
	ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]
	# load model and tokenizer
	tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base', do_lower_case=True, clean_text=False)


	## init dataset and bert model
	context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
	response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
	concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_response_len=args.max_response_length, max_contexts_len=args.max_contexts_length)

	print('Train dir:', args.train_dir)
	print('Output dir:', args.output_dir)

	if not args.eval:
		train_dataset = SelectionDataset(os.path.join(args.train_dir, 'train_42.txt'),
										 context_transform, response_transform, concat_transform, sample_cnt=None, mode=args.architecture)
		val_dataset = SelectionDataset(os.path.join(args.train_dir, 'dev_42.txt'),
									   context_transform, response_transform, concat_transform, sample_cnt=4000, mode=args.architecture)
		train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=0)
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
	else:
		val_dataset = SelectionDataset(args.test_file, context_transform, response_transform, concat_transform, sample_cnt=None, mode=args.architecture)

	val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=val_dataset.batchify_join_str, shuffle=False, num_workers=0)


	epoch_start = 1
	global_step = 0
	best_eval_loss = float('inf')
	best_test_loss = float('inf')

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
	shutil.copyfile(os.path.join(args.bert_model, 'config.json'), os.path.join(args.output_dir, 'config.json'))
	log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')
	print(args, file=log_wf)

	state_save_path = os.path.join(args.output_dir, '{}_{}_{}_pytorch_model.bin'.format(args.architecture, args.poly_m,args.cuda))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	specter_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))
	if not args.eval:
		previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
		print('Loading parameters from', previous_model_file)
		log_wf.write('Loading parameters from %s' % previous_model_file + '\n')
		model_state_dict = torch.load(previous_model_file, map_location="cpu")
		specter = AutoAdapterModel.from_pretrained("allenai/specter2_base", state_dict=model_state_dict)
		adapter_name = specter.load_adapter("allenai/specter2_classification", source="hf", set_active=True)
		del model_state_dict
	else:
		# specter = AutoAdapterModel.from_config(specter_config)
		previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
		model_state_dict = torch.load(previous_model_file, map_location="cpu")
		specter = AutoAdapterModel.from_pretrained("allenai/specter2_base", state_dict=model_state_dict)
		adapter_name = specter.load_adapter("allenai/specter2_classification", source="hf", set_active=True)
		del model_state_dict
	if args.architecture == 'bi':
		model = BiEncoder(specter_config, bert=specter)
	elif args.architecture == 'cross':
		model = CrossEncoder(specter_config, bert=specter)
	elif args.architecture == 'kdcross':
		model = kdCrossEncoder(specter_config, bert=specter)
	else:
		raise Exception('Unknown architecture.')
	model.resize_token_embeddings(len(tokenizer)) 
	model.to(device)
	
	if args.eval:
		print('Loading parameters from', state_save_path)
		model.load_state_dict(torch.load(state_save_path))
		pred_running_model(val_dataloader, out_file=os.path.join(args.output_dir, f'prediction_{args.cuda}_{args.architecture}.txt'))#这里
		exit()

	no_decay = ["bias", "LayerNorm.weight"]
	
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)
	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	print_freq = args.print_freq//args.gradient_accumulation_steps
	eval_freq = 1000
	eval_freq = eval_freq//args.gradient_accumulation_steps
	print('Print freq:', print_freq, "Eval freq:", eval_freq)

	# 初始化参数队列和教师模型begin
	k = 5
	kd_freq = args.kd_freq
	checkpoint_queue = collections.deque(maxlen=k)  # k是队列大小，由您确定
	teacher_model = copy.deepcopy(model)
	for param in teacher_model.parameters():
		param.requires_grad = False

	# 初始化参数队列和教师模型end
	for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
		tr_loss = 0
		nb_tr_steps = 0
		with tqdm(total=len(train_dataloader)//args.gradient_accumulation_steps) as bar:
			for step, batch in enumerate(train_dataloader):
				model.train()
				optimizer.zero_grad()
				batch = tuple(t.to(device) for t in batch)
				if args.architecture == 'cross':
					text_token_ids_list_batch, text_input_masks_list_batch,text_segment_ids_list_batch, labels_batch = batch
					loss = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch)
					loss = loss / args.gradient_accumulation_steps
				#*****************************************************
				elif args.architecture == 'kdcross':
					text_token_ids_list_batch, text_input_masks_list_batch,text_segment_ids_list_batch, labels_batch = batch
					if global_step <= kd_freq:
						score = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch,labels_batch)
						loss =  score[5][0] * score[0] + score[5][1] * score[1] + score[5][2] * score[2] + 1 * score[3]
						loss = loss / args.gradient_accumulation_steps
					else:
						score = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch,labels_batch)
						loss =  score[5][0]*score[0] + score[5][1] * score[1] + score[5][2] * score[2] + 1 * score[3]
						score_teacher = teacher_model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch,labels_batch)
						kd_loss = 0
						kd_loss += score[5][0] * F.mse_loss(score[0], score_teacher[0])
						kd_loss += score[5][1] * F.mse_loss(score[1], score_teacher[1])
						kd_loss += score[5][2] * F.mse_loss(score[2], score_teacher[2])
						kd_loss += 1 * F.mse_loss(score[3], score_teacher[3])

						loss = loss / args.gradient_accumulation_steps + 0.3 * (kd_loss / args.gradient_accumulation_steps)
				#*****************************************************
				else:
					context_token_ids_list_batch, context_input_masks_list_batch, response_token_ids_list_batch, response_input_masks_list_batch,  labels_batch = batch
					loss = model(context_token_ids_list_batch, context_input_masks_list_batch, response_token_ids_list_batch, response_input_masks_list_batch,labels_batch)
					loss = loss / args.gradient_accumulation_steps
				if args.fp16:
					with amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()
				
				tr_loss += loss.item()

				if (step + 1) % args.gradient_accumulation_steps == 0:
					if args.fp16:
						torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
					else:
						torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
					nb_tr_steps += 1
					optimizer.step()
					scheduler.step()
					model.zero_grad()
					global_step += 1
					#*********************
					# 每kd_freq/k步保存一次模型参数到队列
					if global_step>= (kd_freq//k)  and  global_step % (kd_freq // k) == 0:
						checkpoint_queue.append({name: param.clone() for name, param in model.named_parameters()})

						# 如果队列已满，则计算平均参数并更新教师模型
						if len(checkpoint_queue) == k:
							avg_params = {name: sum(param[name] for param in checkpoint_queue) / k for name in
										  checkpoint_queue[0]}

							teacher_model.load_state_dict(avg_params, strict=False)
					# *********************
					if nb_tr_steps and nb_tr_steps % print_freq == 0:
						bar.update(min(print_freq, nb_tr_steps))
						time.sleep(0.02)
						print(global_step, tr_loss / nb_tr_steps)
						log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

					if global_step and global_step % eval_freq == 0:
						val_result = eval_running_model(val_dataloader)
						print('Global Step %d VAL res:\n' % global_step, val_result)
						log_wf.write('Global Step %d VAL res:\n' % global_step)
						log_wf.write(str(val_result) + '\n')
						if args.architecture == 'kdcross':
							print(score[4][0],score[4][1],score[4][2])
							print(score[5][0],score[5][1],score[5][2])
						if val_result['eval_loss'] < best_eval_loss:
							best_eval_loss = val_result['eval_loss']
							val_result['best_eval_loss'] = best_eval_loss
							# save model
							print('[Saving at]', state_save_path)
							log_wf.write('[Saving at] %s\n' % state_save_path)
							torch.save(model.state_dict(), state_save_path)
				log_wf.flush()

		# add a eval step after each epoch
		val_result = eval_running_model(val_dataloader)
		print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
		log_wf.write('Global Step %d VAL res:\n' % global_step)
		log_wf.write(str(val_result) + '\n')

		if val_result['eval_loss'] < best_eval_loss:
			best_eval_loss = val_result['eval_loss']
			val_result['best_eval_loss'] = best_eval_loss
			# save model
			print('[Saving at]', state_save_path)
			log_wf.write('[Saving at] %s\n' % state_save_path)
			torch.save(model.state_dict(), state_save_path)
		print(global_step, tr_loss / nb_tr_steps)
		log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
