import pandas as pd 
import numpy as np
dataset_name = 'atis'

all_labels = pd.read_csv(os.path.join(dataset_name, "train.tsv"), sep="\t")
labels = np.unique(np.array(all_labels['label']))
np.save('labels.npy', labels)