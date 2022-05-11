from cProfile import label
from torch.utils.data import Dataset
from torchvision import datasets
import pickle
import os
import h5py
import numpy as np
import json

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def _load_dataset(dataroot, img_id2val):
    question_path = os.path.join(
        dataroot, "Questions", 'v2_OpenEnded_mscoco_train2014_questions.json')
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', 'knowanswer_target.pkl')
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question['question_id'], answer['question_id'])
        assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    test = question['question'][0]
    return entry

class VQADataset(Dataset):
    # passing hf file handler as a parameter to be closed by the calling code 
    def __init__(self, hf, name = 'train', dataroot='../data') -> None:
        super().__init__()
        # 2 posibile raspunsuri (known answer, unknown answer)
        self.num_ans_candidates = 2

        # folosit pentru a putea incarca feature-urile pozei dupa img_id
        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot,"preprocessed_img_features", '%s36_imgid2idx.pkl' % name), 'rb'))
        print('loading features from h5 file')
        

        h5_path = os.path.join(dataroot,"preprocessed_img_features", '%s36.hdf5' % name)
        # with h5py.File(h5_path, 'r') as hf:
        #     # self.features = np.array(hf.get('image_features'))
        #     # self.spatials = np.array(hf.get('spatial_features'))
        #     # modificat ca sa nu forteze incarcarea de pe disc in memorie
        #     self.features = hf['image_features']

        self.features = hf['image_features']

        self.entries = _load_dataset(dataroot, self.img_id2idx)
        print(f"dataset size is : {len(self.entries)}")

    def __len__(self) -> int:
        return len(self.entries)
    def __getitem__(self, index: int):
        entry = self.entries[index]
        features = self.features[entry['image']]

        question = entry['question']
        answer = entry['answer']
        label = answer['label']

        return features, question, label
