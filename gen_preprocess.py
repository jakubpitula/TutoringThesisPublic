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


def train_test_split(split, path):  # we do the same split for train/validation within the training set
    metadata = pd.read_csv(path + "/teacherStudentChatroomCorpusPublicMetadata.csv")
    cefr_groups_meta = metadata.groupby('student.cefr.level')

    cefr_groups = {}
    for group, contents in cefr_groups_meta:
        cefr_groups[group] = {
            'train': [],
            'validation': [],
            'test': []
        }

        train_split = pd.DataFrame(contents[:][:int(split * len(contents))])
        validation_split = pd.DataFrame(train_split[:][int(split * len(train_split)):])
        train_split = pd.DataFrame(train_split[:][:int(split * len(train_split))])
        test_split = pd.DataFrame(contents[:][int(split * len(contents)):])

        for chat_num, l1 in zip(train_split['chat.num'], train_split['student.L1']):
            cefr_groups[group]['train'].append({
                'chat_num': chat_num,
                'cefr': group,
                'l1': l1
            })

        for chat_num, l1 in zip(validation_split['chat.num'], validation_split['student.L1']):
            cefr_groups[group]['validation'].append({
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

    return pd.DataFrame([res for group in cefr_groups.values() for res in group['train']]), \
        pd.DataFrame([res for group in cefr_groups.values() for res in group['validation']]), \
        pd.DataFrame([res for group in cefr_groups.values() for res in group['test']])


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


def get_training_turns(chatrooms, dialogues_train, dialogues_val):
    chatrooms_train = []
    chatrooms_val = []

    for index, dialogue in dialogues_train.iterrows():
        chatrooms_train.append({
            "dialog": chatrooms[dialogue['chat_num'] - 2],
            "cefr": dialogue['cefr'],
            "l1": dialogue['l1']
        })
    for index, dialogue in dialogues_val.iterrows():
        chatrooms_val.append({
            "dialog": chatrooms[dialogue['chat_num'] - 2],
            "cefr": dialogue['cefr'],
            "l1": dialogue['l1']
        })

    training_dialogues = []
    val_dialogues = []

    for chat in chatrooms_train:
        chat_dialog = chat['dialog']
        chat_cefr = chat['cefr']
        chat_l1 = chat['l1']

        chat_dropped = chat_dialog.dropna(subset=['seq.type'])
        chat_dropped = chat_dropped.loc[chat_dropped['role'].str.contains('teacher')]

        scaffolding_indexes = chat_dropped.loc[chat_dropped['seq.type'].str.contains('scaffolding'), ['turn.number']]

        turns = get_turns(chat_dialog, scaffolding_indexes)
        training_dialogues.append({
            'dialog': turns,
            'cefr': chat_cefr,
            'l1': chat_l1
        })

    for chat in chatrooms_val:
        chat_dialog = chat['dialog']
        chat_cefr = chat['cefr']
        chat_l1 = chat['l1']

        chat_dropped = chat_dialog.dropna(subset=['seq.type'])
        chat_dropped = chat_dropped.loc[chat_dropped['role'].str.contains('teacher')]

        scaffolding_indexes = chat_dropped.loc[chat_dropped['seq.type'].str.contains('scaffolding'), ['turn.number']]

        turns = get_turns(chat_dialog, scaffolding_indexes)
        val_dialogues.append({
            'dialog': turns,
            'cefr': chat_cefr,
            'l1': chat_l1
        })
    return training_dialogues, val_dialogues


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
            You are an English language teacher in a classroom dialogue with a student.\n
            The student is on the {sample['cefr']} English level in CEFR scale and {sample['l1']} is their mother tongue.\n
            Based on the provided context, provide a helpful pedagogical feedback to the student.\n
            Your response must aim to enhance the student's English skills.\n
            Do not engage in a casual conversation with the student.\n
            Even if the student asks you a casual question, include an English teaching element in your response.\n
            Be concise in your response. Use around 15 words.\n
            <</SYS>>
            
            {prompt_complete}
        """

        prompts.append(prompt)

    return prompts


def max_scaffolding_length(chatrooms):
    chatrooms_all = pd.concat(chatrooms, axis=0, ignore_index=True)
    scaffolding_utts = chatrooms_all[(chatrooms_all['role'].str.contains('teacher')) &
                                     (chatrooms_all['seq.type'].str.contains('scaffolding'))
                                     ]
    return np.max(scaffolding_utts['nWords'])


def gen_preprocess(split, sample_len=9):
    # 0. Import the data

    path = 'TeacherStudentChatroomCorpus_v2/public'
    filenames = glob.glob(path + "/*.tsv")

    chatrooms = []
    for filename in filenames:
        chatrooms.append(pd.read_csv(filename, sep='\t'))

    # 1. Make a CEFR-level balanced train/test split (80/20)

    dialogues_train, dialogues_val, dialogues_test = train_test_split(split, path)
    # print(len(dialogues_train), len(dialogues_test))

    # 2. Extract scaffolding utterances and context samples for the test set.

    scaffolding_context_samples = get_scaffolding_context_samples(chatrooms, dialogues_test, sample_len)
    # print(len(get_scaffolding_context_samples(chatrooms, dialogues_train, sample_len)))
    # 3. Create llama prompts

    prompts = make_prompts(scaffolding_context_samples)

    # 4. Calculate the mean length of tutor scaffolding utterances to set max gen length

    max_scaff_len = max_scaffolding_length(chatrooms)

    # 5. Get training turns for fine-tuning

    training_turns, validation_turns = get_training_turns(chatrooms, dialogues_train, dialogues_val)

    return {
        'prompts': prompts,
        'samples': scaffolding_context_samples,
        'max_gen_length': math.ceil(max_scaff_len),
        'train': training_turns,
        'validation': validation_turns
    }


gen_preprocess(0.8, 9)
