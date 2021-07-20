import pandas as pd
import numpy as np

test = pd.read_csv("train.tsv", sep="\t")
labels = np.unique(np.array(test['label']))
np.save('labels.npy', labels)