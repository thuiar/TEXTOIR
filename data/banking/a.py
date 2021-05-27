import numpy as np
import pandas as pd
# a = pd.read_csv('test.tsv', sep="\t")
# b = a['label']
# c = np.unique(np.array(b))
# dataframe = pd.DataFrame({'label': c})
# dataframe.to_csv("labels.csv", index=False)
a = np.load('labels.npy', allow_pickle=True)
print(len(a))