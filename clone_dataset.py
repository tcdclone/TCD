import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CodeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, 
                 tokenizer, 
                 max_input_length=900, 
                 max_output_length=900):
        self.dataframe = dataframe
        self.labels = dataframe['label'].unique()
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Tokenize all solutions and store them
        self.tokenized_solutions = []
        for idx, row in dataframe.iterrows():
            tokenized_solution = tokenizer(row['flines'], truncation=True, max_length=max_input_length)
            self.tokenized_solutions.append(tokenized_solution)

    def __len__(self):
        return len(self.dataframe)
    
    def pad_tokens(self, token_list, pad_value=None):
        if pad_value is None:
            pad_value = self.tokenizer.pad_token_id
        if len(token_list) > self.max_input_length: 
            if token_list[-1] == self.tokenizer.eos_token_id:
                return token_list[:self.max_input_length-1] + [self.tokenizer.eos_token_id]
            else:
                return token_list[:self.max_input_length]
        else:
            left_pad = self.max_input_length - len(token_list)
            return token_list + [pad_value]*left_pad
    
    def add_prompt_to_output(self, prompt, idx):
        """
        prompt do not cal loss
        """
        prompt_tokens = self.tokenizer(prompt).input_ids[:-1] # skip eos
        prompt_len = len(prompt_tokens)
        code_tokens = self.tokenized_solutions[idx]['input_ids'][1:] # skip cls
        code_mask = self.tokenized_solutions[idx]['attention_mask'][1:] # 

        input_ids = prompt_tokens + code_tokens
        attention_mask = [1]*prompt_len + code_mask
        output_ids = [-100]*prompt_len + code_tokens # prompt donot calculate loss
        output_ids = output_ids[1:]
        return {"input_ids": self.pad_tokens(input_ids),
                "attention_mask": self.pad_tokens(attention_mask, pad_value=0),
                "output_ids": self.pad_tokens(output_ids, pad_value=-100)
        }
    
    def get_data_decoder_prompt(self,label_idx, source_idx, target_idx, prompt):
        
        source_code = self.tokenized_solutions[source_idx]
        encoder_input_ids = source_code['input_ids']
        encoder_attention_mask = source_code['attention_mask']
        encoder_input_ids = self.pad_tokens(encoder_input_ids)
        encoder_attention_mask = self.pad_tokens(encoder_attention_mask, 0)

        decoder_output = self.add_prompt_to_output(prompt, target_idx)

        return {
            'label': label_idx,
            'encoder_input_ids': encoder_input_ids,
            'encoder_attention_mask': encoder_attention_mask,
            'decoder_input_ids': decoder_output['input_ids'],
            'decoder_attention_mask': decoder_output['attention_mask'],
            'decoder_output_ids': decoder_output['output_ids']
        }
    
        
    def __getitem__(self, idx):
        source_sample = self.dataframe.iloc[idx]

        # get int label 
        label_str = source_sample['label']
        label_idx = self.label_to_index[label_str]

        # Get another sample with the same label
        same_label_samples = self.dataframe[self.dataframe['label'] == label_str]
        sample_index = same_label_samples.sample(n=1).index
        target_idx = sample_index[0]
        target_sample = self.dataframe.iloc[target_idx]
        # get languages of two samples
        source_l, target_l = source_sample['lan'], target_sample['lan']
        if source_l == target_l:
            prompt = reverse_prompt = "Rewrite: "
        else:
            prompt = f"Translate from {source_l} to {target_l}"
            reverse_prompt = f"Translate from {target_l} to {source_l}"
    
        return [
            self.get_data_decoder_prompt(label_idx, source_idx=idx, 
                                         target_idx=target_idx, prompt=prompt),
            self.get_data_decoder_prompt(label_idx, source_idx=target_idx, 
                                         target_idx=idx, prompt=reverse_prompt)
        ]


# Custom collate_fn to process batch data
def collate_fn(batch):
    labels, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, decoder_output_ids = [], [], [], [], [], []
    for item1, item2 in batch:
        labels.append(item1['label'])
        labels.append(item2['label'])
        encoder_input_ids.append(item1['encoder_input_ids'])
        encoder_input_ids.append(item2['encoder_input_ids'])
        encoder_attention_mask.append(item1['encoder_attention_mask'])
        encoder_attention_mask.append(item2['encoder_attention_mask'])
        decoder_input_ids.append(item1['decoder_input_ids'])
        decoder_input_ids.append(item2['decoder_input_ids'])
        decoder_attention_mask.append(item1['decoder_attention_mask'])
        decoder_attention_mask.append(item2['decoder_attention_mask'])
        decoder_output_ids.append(item1['decoder_output_ids'])
        decoder_output_ids.append(item2['decoder_output_ids'])
    
    return {
        'labels': torch.tensor(labels, dtype=torch.long),
        'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
        'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
        'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
        'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
        'decoder_output_ids': torch.tensor(decoder_output_ids, dtype=torch.long)
    }   

# Function to load the dataset and return DataLoader
def load_dataset(file_name, tokenizer):
    all_data = pd.read_pickle(file_name)

    # all_data = all_data.sample(n=100) # only for pipeline test 
    if 'test' in file_name:
        all_data = all_data.sample(frac=0.4)
    all_data.reset_index(drop=True, inplace=True)
    dataset = CodeDataset(all_data, tokenizer)
    return dataset

if __name__ == "__main__": # only for test
    from transformers import AutoTokenizer
    model_name = "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset('test', tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn)
    for b in dataloader:
        print(b)
