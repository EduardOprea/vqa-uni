import base64
import sys
import numpy as np
import csv
import torch

from transformers import BertTokenizerFast, VisualBertForQuestionAnswering

from vqa_model import VQAModel


maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = '../data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
train_data_file = '../data/preprocessed_img_features/train36.hdf5'
val_data_file = '../data/preprocessed_img_features/val36.hdf5'
train_indices_file = '../data/preprocessed_img_features/train36_imgid2idx.pkl'
val_indices_file = '../data/preprocessed_img_features/val36_imgid2idx.pkl'
train_ids_file = '../data/preprocessed_img_features/train_ids.pkl'
val_ids_file = '../data/preprocessed_img_features/val_ids.pkl'

def load_feature_for_image(infile):
    print("reading tsv...")
    count = 0
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            count = count + 1
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(bytes(item['boxes'], encoding='utf-8')),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            features = np.frombuffer(
                    base64.decodestring(bytes(item['features'], encoding='utf-8')),
                    dtype=np.float32).reshape((item['num_boxes'], -1))

            
            if count == 1:
                break
    return features, image_id


features, image_id = load_feature_for_image('../data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv')
test = False

features_tensor = torch.Tensor([features])
print(features_tensor.shape)
print(features.shape)

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")


know_answer_vqa = VQAModel(visualbert_vqa.visual_bert, visualbert_vqa.cls, visualbert_vqa.config)

test_questions_for_url2 = [
    "Where is the cat?",
    "What is near the disk?",
    "What is the color of the table?",
    "What is the color of the cat?",
    "What is the shape of the monitor?",
]


for test_question in test_questions_for_url2:
    test_question = [test_question]

    inputs = bert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # output_vqa = visualbert_vqa(
    #     input_ids=inputs.input_ids,
    #     attention_mask=inputs.attention_mask,
    #     visual_embeds=features,
    #     visual_attention_mask=torch.ones(features.shape[:-1]),
    #     token_type_ids=inputs.token_type_ids,
    #     output_attentions=False,
    # )

    
    output_vqa = know_answer_vqa(input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features_tensor,
            visual_attention_mask=torch.ones(features_tensor.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False)
    # get prediction
    pred_vqa = output_vqa[1].argmax(-1)
    print("Question:", test_question)
    print("prediction from VisualBert VQA:", pred_vqa)
