import numpy as np
import torch
import os
import csv
import sys
import logging
from transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


class BERT_Loader_NEG:
    def __init__(self, args, base_attrs, logger_name = 'Detection'):
        self.logger = logging.getLogger(logger_name)
        self.train_examples, self.train_labeled_examples, self.train_unlabeled_examples  = get_examples(args, base_attrs, 'train')
        self.logger.info("Number of labeled training samples = %s", str(len(self.train_labeled_examples)))
        self.logger.info("Number of unlabeled training samples = %s", str(len(self.train_unlabeled_examples)))

        self.eval_examples = get_examples(args, base_attrs, 'eval')
        self.logger.info("Number of evaluation samples = %s", str(len(self.eval_examples)))

        self.test_examples = get_examples(args, base_attrs, 'test')
        self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))

        self.neg_examples = get_examples(args, base_attrs, 'neg')
        self.logger.info("Number of neg samples = %s", str(len(self.neg_examples)))

        self.base_attrs = base_attrs
        self.init_loader(args)

    def init_loader(self, args):
        self.train_labeled_loader = get_loader(self.train_labeled_examples, args, self.base_attrs['label_list'], 'train_labeled')
        self.train_unlabeled_loader = get_loader(self.train_unlabeled_examples, args, self.base_attrs['label_list'], 'train_unlabeled')
        self.eval_loader = get_loader(self.eval_examples, args, self.base_attrs['label_list'], 'eval')
        self.test_loader = get_loader(self.test_examples, args, self.base_attrs['label_list'], 'test')
        self.neg_loader = get_loader(self.neg_examples, args, self.base_attrs['label_list'], 'neg')
        self.num_train_examples = len(self.train_labeled_examples)

def get_examples(args, base_attrs, mode):

    processor = DatasetProcessor()
    ori_examples = None
    if mode != 'neg':
        ori_examples = processor.get_examples(base_attrs['data_dir'], mode)

    if mode == 'train':

        labeled_examples, unlabeled_examples = [], []
        for example in ori_examples:
            if (example.label in base_attrs['known_label_list']) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                labeled_examples.append(example)
            else:
                example.label = base_attrs['unseen_label']
                unlabeled_examples.append(example)

        return ori_examples, labeled_examples, unlabeled_examples

    elif mode == 'eval':

        examples = []
        for example in ori_examples:
            if (example.label in base_attrs['known_label_list']):
                examples.append(example)
        
        return examples
    
    elif mode == 'test':

        examples = []
        for example in ori_examples:
            if (example.label in base_attrs['label_list']) and (example.label is not base_attrs['unseen_label']):
                examples.append(example)
            else:
                example.label = base_attrs['unseen_label']
                examples.append(example)
        return examples
        
    
    # add negative set
    elif mode == 'neg':
        examples = []
        neg_examples = processor.get_examples(base_attrs['data_dir_neg'], "neg")
        for i, example in enumerate(neg_examples):
            example.label = base_attrs['unseen_label']
            examples.append(example)
        examples.extend(neg_examples)

        return examples

def get_loader(examples, args, label_list, mode):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)    
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if mode == 'train_unlabeled':
        label_ids = torch.tensor([-1 for f in features], dtype=torch.long)
    else:
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    if mode == 'train_labeled':   
        dataloader = DataLoader(datatensor, shuffle = True, batch_size = args.train_batch_size)    

    else:
        sampler = SequentialSampler(datatensor)

        if mode == 'train_unlabeled':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size)    
        elif mode == 'test':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.test_batch_size)    
        elif mode == 'neg':
            dataloader = DataLoader(datatensor, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    
    return dataloader

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'neg':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "squad.tsv")), "neg"
            )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    if label_list is not None:
        for i, label in enumerate(label_list):
            label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
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