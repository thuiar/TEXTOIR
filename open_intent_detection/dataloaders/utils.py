import csv
import sys
import os
import numpy as np
import torch

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
    
    
def get_examples(args, base_attrs, mode):

    processor = DatasetProcessor()
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
    
class KCCLDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, k_pos=0, n_neg=0,
                 mode='train', neg_label=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label2sid = dict()    
        self.k = k_pos
        self.n = n_neg
        self.mode = mode
        self.neg_label = neg_label

        if k_pos > 0:
            for sid, i in enumerate(label_ids.detach().cpu().numpy()):
                if i not in self.label2sid:
                    self.label2sid[i] = [sid]
                else:
                    self.label2sid[i].append(sid)

    def generate_postive_sample(self, label, self_index):
        if self.k > 0:
            index_list = [ind for ind in self.label2sid[label] if ind != self_index]
            return np.random.choice(index_list, size=self.k, replace=False)
        else:
            return None

    def generate_negtive_sample(self, label):
        if self.n > 0:
            index_list = []
            for key, value in self.label2sid.items():
                if key != label:
                    index_list += value
            return np.random.choice(index_list, size=self.n, replace=False)
        else:
            return None

    def __getitem__(self, idx):
        sid = self.generate_postive_sample(self.label_ids[idx].item(), idx)
        if self.n > 0:
            nid = self.generate_negtive_sample(self.label_ids[idx].item())
            sids = np.append([idx], sid)
            sids = np.append(sids, nid)
        else:
            sids = np.append([idx], sid)
        input_ids = self.input_ids[sids]    
        input_mask = self.input_mask[sids]
        segment_ids = self.segment_ids[sids]
        label_ids = self.label_ids[sids]
        return {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_ids}

    def __len__(self):
        return len(self.label_ids)