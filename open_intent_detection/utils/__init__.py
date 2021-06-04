import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import random
import csv
import sys
import math
import json
import importlib
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# from torch.nn.utils import weight_norm
