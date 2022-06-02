import os
import pickle
import json

train_answer_file = '../data/Annotations/v2_mscoco_train2014_annotations.json'
train_answers = json.load(open(train_answer_file))['annotations']
train_answers = sorted(train_answers, key=lambda x: x['question_id'])
answer_path = os.path.join('../data', 'Annotations', f'train_target.pkl')
answers = pickle.load(open(answer_path, 'rb'))
answers = sorted(answers, key=lambda x: x['question_id'])
lab2answer_path = os.path.join('../data', 'cache', f'trainval_label2ans.pkl')
lab2answer = pickle.load(open(lab2answer_path, 'rb'))
test = False