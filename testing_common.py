import json
from flask import jsonify


train_answer_file = '../data/Annotations/v2_mscoco_train2014_annotations.json'
train_answers = json.load(open(train_answer_file))['annotations']
ceva = "sta"