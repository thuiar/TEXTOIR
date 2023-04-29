import torch
import random
from typing import Any, Dict, Union


def create_negative_dataset(train_dataloader):

    list = []
    for step, inputs in enumerate(train_dataloader):
        input_ids, input_masks, segment_ids, label_ids = inputs
        input_ids = input_ids.tolist()
        input_masks = input_masks.tolist()
        segment_ids = segment_ids.tolist()
        label_ids = label_ids.tolist()

        for i in range(len(input_ids)-1):
            input_id = input_ids[i]
            input_mask = input_masks[i]
            segment_id = segment_ids[i]
            label_id = label_ids[i]
            batch_dict = {"labels": label_id, "input_ids": input_id, "token_type_ids": segment_id,
                                "attention_mask": input_mask}
            list.append(batch_dict)
            
    negative_dataset = {}

    for line in list:
        label = int(line["labels"])
        inputs = line
        inputs.pop("labels")
        if label not in negative_dataset.keys():
            negative_dataset[label] = [inputs]
        else:
            negative_dataset[label].append(inputs)

    return negative_dataset


def generate_positive_sample(negative_data, args, label: torch.Tensor):
    positive_num = args.positive_num # 3
    # positive_num = 16
    positive_sample = []
    for index in range(label.shape[0]):
        input_label = int(label[index])
        positive_sample.extend(random.sample(negative_data[input_label], positive_num))

    return list_item_to_tensor(positive_sample)

def _prepare_inputs(device, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
                    
    return inputs


def list_item_to_tensor(inputs_list):
    batch_list = {}
    for key, value in inputs_list[0].items():
        batch_list[key] = []
    for inputs in inputs_list:
        for key, value in inputs.items():
            batch_list[key].append(value)

    batch_tensor = {}
    for key, value in batch_list.items():
        batch_tensor[key] = torch.tensor(value)
    return batch_tensor
