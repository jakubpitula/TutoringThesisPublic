import copy
import itertools
from datasets import Dataset
import pandas as pd

from gen_preprocess import gen_preprocess
from transformers import LlamaTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"


# tok = LlamaTokenizer.from_pretrained("../llama/llama-2-13b-chat-hf")


def tokenize_dialog(dialog, tokenizer):
    dial = dialog['dialog']
    cefr = dialog['cefr']
    l1 = dialog['l1']
    sys_prompt = f"""You are an English language teacher in a classroom dialogue with a student.\n
                The student is on the {cefr} English level in CEFR scale and {l1} is their mother tongue.\n"""

    prompt_tokens = [f"{tokenizer.bos_token}{B_INST} {B_SYS}\n"
                     f"{sys_prompt}{E_SYS}\n\n"
                     f"{dial[0]['content'].strip()} {E_INST} "] + \
                    [f"{tokenizer.bos_token}{B_INST} {prompt['content'].strip()} {E_INST} " for prompt in dial[2::2] if prompt]
    answer_tokens = [f"{answer['content'].strip()} {tokenizer.eos_token}" for answer in dial[1::2] if answer]

    dialog_tokens = ''.join(list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens))))
    # # Add labels, convert prompt token to -100 in order to ignore in loss function
    # labels_tokens = [len(c) * [-100, ] if i % 2 == 0 else c for i, c in enumerate(dialog_tokens)]
    #
    # combined_tokens = {
    #     "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
    #     "labels": list(itertools.chain(*(t for t in labels_tokens))),
    # }
    # print(f"""Length:
    # prompt tokens = {len(prompt_tokens)}
    # answer tokens = {len(answer_tokens)}
    # label tokens = {len(labels_tokens)}
    # input ids = {len(combined_tokens['input_ids'])}
    # labels = {len(combined_tokens['labels'])}""")
    # return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))

    return {"dialog": dialog_tokens}

def get_scaffolding_samples(turns, sample_len):
    dialog = turns['dialog']
    scaffolding_context_samples = []
    scaffolding_turns = dialog.loc[(dialog['is_scaffolding']) & (dialog.index >= sample_len)]
    # Extract preceding context samples.
    for index, scaffolding_turn in scaffolding_turns.iterrows():
        sample = {'dialog': pd.DataFrame.from_dict({
            'turn': [turn for turn in dialog.loc[index - sample_len:index, 'turn'].values],
            'speaker': [speaker for speaker in dialog.loc[index - sample_len:index, 'speaker'].values]
        }), 'cefr': turns['cefr'], 'l1': turns['l1']}
        # keys = list(sample['context'].keys())
        # values = list(sample['context'].values())
        #
        # sample['context'] = [dict(zip(keys, value)) for value in zip(*values)]
        # sample['scaffolding'] = scaffolding_turn['turn']
        scaffolding_context_samples.append(sample)
    return scaffolding_context_samples


def get_custom_dataset(dataset_config, tokenizer, split):
    preprocess_results = gen_preprocess(split=split)
    dataset_train = preprocess_results['train']
    dataset_eval = preprocess_results['validation']

    dataset_train = list(
        itertools.chain.from_iterable([get_scaffolding_samples(dialog, sample_len=9) for dialog in dataset_train]))
    dataset_eval = list(
        itertools.chain.from_iterable([get_scaffolding_samples(dialog, sample_len=9) for dialog in dataset_eval]))
    # print(len(dataset_train), len(dataset_eval))
    for i in range(len(dataset_train)):
        dataset_train[i]['dialog'] = dataset_train[i]['dialog'].reset_index(drop=True)
        dataset_train[i]['dialog'] = dataset_train[i]['dialog']['turn'].values

    dataset_train = Dataset.from_pandas(pd.DataFrame(dataset_train))

    for i in range(len(dataset_eval)):
        dataset_eval[i]['dialog'] = dataset_eval[i]['dialog'].reset_index(drop=True)
        dataset_eval[i]['dialog'] = dataset_eval[i]['dialog']['turn'].values

    dataset_eval = Dataset.from_pandas(pd.DataFrame(dataset_eval))

    print(list(dataset_train.features))

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread['dialog']):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog, "cefr": thread['cefr'], "l1": thread['l1']}

    dataset_train = dataset_train.map(lambda x: to_dialog(x), remove_columns=list(dataset_train.features))
    dataset_eval = dataset_eval.map(lambda x: to_dialog(x), remove_columns=list(dataset_eval.features))

    # print(dataset_train['dialog'], dataset_train)

    dataset_train = dataset_train.map(lambda x: tokenize_dialog(x, tokenizer),
                                      remove_columns=list(dataset_train.features))
    dataset_eval = dataset_eval.map(lambda x: tokenize_dialog(x, tokenizer), remove_columns=list(dataset_eval.features))

    # print(dataset_train.data)
    # print(dataset_train.features)
    # print(dataset_eval.features)
    # print(dataset_eval.data)

    return {
        "train": dataset_train,
        "eval": dataset_eval
    }

# get_custom_dataset('', 'tok', 0.8)
