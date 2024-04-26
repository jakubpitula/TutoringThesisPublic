import glob
import math

import pandas as pd
import numpy as np


def get_turns(chat, scaffolding_indexes):
    turns = {
        'turn': [],
        'speaker': [],
        'is_scaffolding': []
    }

    curr_turn = ""
    curr_speaker = chat['role'][0]
    is_scaffolding = False
    for index, utt in chat.iterrows():
        if utt['role'] != curr_speaker:
            turns['turn'].append(curr_turn)
            turns['speaker'].append(curr_speaker)
            turns['is_scaffolding'].append(is_scaffolding)

            curr_turn = ""  # reset when new speaker is speaking
            is_scaffolding = False  # reset the scaffolding flag too

        curr_turn += utt['anonymised']
        if curr_turn[-1] not in ('.', '!', '?'):
            curr_turn += '.'
        curr_turn += ' '

        curr_speaker = utt['role']

        if not is_scaffolding and utt['turn.number'] in scaffolding_indexes.values:
            is_scaffolding = True  # we find a scaffolding utterance - the entire turn is considered scaffolding

    # add the last utterance
    turns['turn'].append(curr_turn)
    turns['speaker'].append(curr_speaker)
    turns['is_scaffolding'].append(is_scaffolding)

    return pd.DataFrame.from_dict(turns)


def train_test_split(split, path):
    metadata = pd.read_csv(path + "/teacherStudentChatroomCorpusPublicMetadata.csv")
    cefr_groups_meta = metadata.groupby('student.cefr.level')

    cefr_groups = {}
    for group, contents in cefr_groups_meta:
        cefr_groups[group] = {
            'train': [],
            'test': []
        }

        train_split = pd.DataFrame(contents[:][:int(split * len(contents))])
        test_split = pd.DataFrame(contents[:][int(split * len(contents)):])

        for chat_num, l1 in zip(train_split['chat.num'], train_split['student.L1']):
            cefr_groups[group]['train'].append({
                'chat_num': chat_num,
                'cefr': group,
                'l1': l1
            })

        for chat_num, l1 in zip(test_split['chat.num'], test_split['student.L1']):
            cefr_groups[group]['test'].append({
                'chat_num': chat_num,
                'cefr': group,
                'l1': l1
            })

        # print(f"{group} train: {len(cefr_groups[group]['train'])}\n")
        # print(f"{group} train: {len(cefr_groups[group]['test'])}\n")

    return pd.DataFrame([res for group in cefr_groups.values() for res in group['train']]), pd.DataFrame(
        [res for group in
         cefr_groups.values() for
         res in group['test']])


def get_scaffolding_context_samples(chatrooms, dialogues_test, sample_len):
    assert sample_len % 2 == 1  # it has to be odd so that the first utterance is student's
    chatroom_nums = dialogues_test['chat_num']
    cefr_levels = dialogues_test['cefr']
    l1s = dialogues_test['l1']

    chatrooms_test = []
    for index, dialogue in dialogues_test.iterrows():
        chatrooms_test.append({
            'chat': chatrooms[dialogue['chat_num'] - 2],
            'cefr': dialogue['cefr'],
            'l1': dialogue['l1']
        })

    scaffolding_context_samples = []

    for chat in chatrooms_test:
        chat_dropped = chat['chat'].dropna(subset=['seq.type'])
        chat_dropped = chat_dropped.loc[chat_dropped['role'].str.contains('teacher')]

        scaffolding_indexes = chat_dropped.loc[chat_dropped['seq.type'].str.contains('scaffolding'), ['turn.number']]

        # Now, I combine utterances into turns (including the scaffolding ones)

        turns = get_turns(chat['chat'], scaffolding_indexes)

        # I always want a context of 10 previous turns, so I discard scaffolding utterances which
        # appear before the 10th turn

        scaffolding_turns = turns.loc[(turns['is_scaffolding']) & (turns.index >= sample_len)]

        # Extract preceding context samples.

        for index, scaffolding_turn in scaffolding_turns.iterrows():
            sample = {'context': {
                'turn': [turn for turn in turns.loc[index - sample_len:index - 1, 'turn'].values],
                'speaker': [speaker for speaker in turns.loc[index - sample_len:index - 1, 'speaker'].values]
            },
                'scaffolding': scaffolding_turn['turn']}

            keys = list(sample['context'].keys())
            values = list(sample['context'].values())

            sample['context'] = [dict(zip(keys, value)) for value in zip(*values)]
            sample['scaffolding'] = scaffolding_turn['turn']
            sample['cefr'] = chat['cefr']
            sample['l1'] = chat['l1']

            scaffolding_context_samples.append(sample)

    return scaffolding_context_samples


# FROM LLAMA GITHUB https://github.com/meta-llama/llama/blob/main/llama/generation.py#L212C12
''' 
 "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
'''


# Hence, I have to assume that the student's utterance comes first and format the data this way.

def make_prompts(samples):
    prompts = []

    for sample in samples:
        prompt_complete = f"{sample['context'][0]['turn']} [/INST]"  # completing the first student utterance
        for turn in sample['context'][1:]:
            if turn['speaker'] == 'teacher':
                prompt_complete += f" {turn['turn']} </s>"
            else:
                prompt_complete += f"<s>[INST] {turn['turn']} [/INST]"

        # CEFR and native
        prompt = f"""
        <s>[INST] <<SYS>>
            You are a second language teacher in a classroom dialogue with a student.\n
            Based on the provided context, helpfully and pedagogically answer the last student’s turn.\n
            Aim to enhance student’s language skills with your response.\n
            <</SYS>>

            {prompt_complete}
        """

        prompts.append(prompt)

    return prompts


def mean_scaffolding_length(chatrooms):
    chatrooms_all = pd.concat(chatrooms, axis=0, ignore_index=True)
    scaffolding_utts = chatrooms_all[(chatrooms_all['role'].str.contains('teacher')) &
                                     (chatrooms_all['seq.type'].str.contains('scaffolding'))
                                     ]
    return np.mean(scaffolding_utts['nWords'])


def gen_preprocess(split, sample_len):
    # 0. Import the data

    path = 'TeacherStudentChatroomCorpus_v2/public'
    filenames = glob.glob(path + "/*.tsv")

    chatrooms = []
    for filename in filenames:
        chatrooms.append(pd.read_csv(filename, sep='\t'))

    # 1. Make a CEFR-level balanced train/test split (80/20)

    dialogues_train, dialogues_test = train_test_split(split, path)
    print(len(dialogues_train), len(dialogues_test))

    # 2. Extract scaffolding utterances and context samples for the test set.

    scaffolding_context_samples = get_scaffolding_context_samples(chatrooms, dialogues_test, sample_len)
    print(len(scaffolding_context_samples))
    # 3. Create llama prompts

    prompts = make_prompts(scaffolding_context_samples)

    # 4. Calculate the mean length of tutor scaffolding utterances to set max gen length

    mean_scaff_len = mean_scaffolding_length(chatrooms)

    return {
        'prompts': prompts,
        'samples': scaffolding_context_samples,
        'max_gen_length': math.ceil(mean_scaff_len)
    }

# preprocess_results = gen_preprocess(split=0.8, sample_len=9)
#
# prompts = preprocess_results['prompts']
# samples = preprocess_results['samples']
# # print(len(samples[567:]))
# print(len(samples))
