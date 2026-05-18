import numpy as np
import torch
import logging
from transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from .utils import InputFeatures, InputExample, get_examples, KCCLDataset
import uuid
import json
import tqdm
import os


class BERT_Loader:
    def __init__(self, args, base_attrs, logger_name = 'Detection'):
        self.logger = logging.getLogger(logger_name)

        self.train_examples, self.train_labeled_examples, self.train_unlabeled_examples  = get_examples(args, base_attrs, 'train')

        self.logger.info("Number of labeled training samples = %s", str(len(self.train_labeled_examples)))
        self.logger.info("Number of unlabeled training samples = %s", str(len(self.train_unlabeled_examples)))

        self.eval_examples = get_examples(args, base_attrs, 'eval')
        self.logger.info("Number of evaluation samples = %s", str(len(self.eval_examples)))

        self.test_examples = get_examples(args, base_attrs, 'test')
        self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
        self.base_attrs = base_attrs
        self.init_loader(args)
        

    def init_loader(self, args):
        
        self.train_labeled_loader = get_loader(self.train_labeled_examples, args, self.base_attrs['label_list'], 'train_labeled', sampler_mode = 'random')
        self.train_unlabeled_loader = get_loader(self.train_unlabeled_examples, args, self.base_attrs['label_list'], 'train_unlabeled', sampler_mode = 'sequential')
        self.eval_loader = get_loader(self.eval_examples, args, self.base_attrs['label_list'], 'eval', sampler_mode = 'sequential')
        self.test_loader = get_loader(self.test_examples, args, self.base_attrs['label_list'], 'test', sampler_mode = 'sequential')
        if 'kccl_k' in args:
            self.k_positive_dataloader = get_loader(self.train_labeled_examples, args, self.base_attrs['label_list'], 'k_positive', sampler_mode = 'random')

        self.num_train_examples = len(self.train_labeled_examples)


def get_loader(examples, args, label_list, mode, sampler_mode = 'sequential'):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if mode == 'train_unlabeled':
        label_ids = torch.tensor([-1 for f in features], dtype=torch.long)
    else:
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    if mode != 'k_positive':
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    else:
        datatensor = KCCLDataset(input_ids, input_mask, segment_ids, label_ids, 0 if mode != 'k_positive' else args.kccl_k, 0 if mode != 'k_positive' else args.neg_num)

    if sampler_mode == 'random':
        sampler = RandomSampler(datatensor)
    elif sampler_mode == 'sequential':
        sampler = SequentialSampler(datatensor)

    if mode in ('train_labeled', 'train_unlabeled', 'k_positive'):
        dataloader = DataLoader(datatensor, sampler = sampler, batch_size = args.train_batch_size)
    elif mode == 'eval':
        dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size)    
    elif mode == 'test':
        dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.test_batch_size)    
    
    return dataloader



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    if label_list is not None:
        for i, label in enumerate(label_list):
            label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a if hasattr(example, 'text_a') else example)

        tokens_b = None
        if hasattr(example, 'text_b') and example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] if label_list is not None else None
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
