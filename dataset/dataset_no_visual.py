from cProfile import label
from torch.utils.data import Dataset, Sampler
from torchvision import datasets
import torch
import pickle
import os
import h5py
import numpy as np
import json


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)






def _load_dataset(dataroot, name = 'train'):
    question_path = os.path.join(
        dataroot, "Questions", f'v2_OpenEnded_mscoco_{name}2014_questions.json')
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    knowanswer_labels_path = os.path.join(dataroot, 'cache', f'knowanswer_{name}_target.pkl')
    knowanswer_labels = pickle.load(open(knowanswer_labels_path, 'rb'))
    knowanswer_labels = sorted(knowanswer_labels, key=lambda x: x['question_id'])

    ans_entropy_path = os.path.join(dataroot, 'cache', f'{name}_qa_entropy.pkl')
    ans_entropies = pickle.load(open(ans_entropy_path, 'rb'))
    ans_entropies = sorted(ans_entropies, key = lambda x : x['question_id'])

    answers_path = os.path.join(
        dataroot, "Annotations", f'v2_mscoco_{name}2014_annotations.json')
    answers = sorted(json.load(open(answers_path))['annotations'],
                       key=lambda x: x['question_id'])
    assert_eq(len(questions), len(knowanswer_labels))
    assert_eq(len(questions), len(answers))
    assert_eq(len(questions), len(ans_entropies))
    entries = []
    for question, label, answer, ans_entropy in zip(questions, knowanswer_labels, answers, ans_entropies):
        assert_eq(question['question_id'], label['question_id'])
        assert_eq(question['image_id'], label['image_id'])
        img_id = question['image_id']
        label['gt_answer'] = answer['multiple_choice_answer']
        label['answers_entropy'] = ans_entropy['answers_entropy']
        label['confidence_score'] = ans_entropy['confidence_score']
        entries.append(_create_entry(question, label))

    return entries

def _create_entry(question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    
    entry = {
        'question_id' : question['question_id'],
        'question'    : question['question'],
        'answer'      : answer}
    
    return entry

class SubsetSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)

class VQANoVisualFeaturesDataset(Dataset):
    # passing hf file handler as a parameter to be closed by the calling code 
    def __init__(self, name = 'train', dataroot='../data') -> None:
        super().__init__()
        # 2 posibile raspunsuri (known answer, unknown answer)
        self.num_ans_candidates = 2

        self.entries = _load_dataset(dataroot, name)
        print(f"dataset size is : {len(self.entries)}")

    def __len__(self) -> int:
        return len(self.entries)
    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['question']
        answer = entry['answer']
        label = answer['label']
        gt_answer = answer['gt_answer']
        entropy = answer['answers_entropy']
        confidence = answer['confidence_score']

        return question, label, gt_answer, entropy, confidence


