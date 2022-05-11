import json
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from typing import Tuple, Union
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from modelling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config
import utils
from vqa_model import VQAModel
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast

class VQAInferenceResult:
    def __init__(self, answer: str, answer_known: bool) -> None:
        self.answer = answer
        self.answer_known = answer_known
        

def inference(model: VQAModel, preprocesser: Preprocess,
             frcnn: GeneralizedRCNN,tokenizer: BertTokenizerFast, image_path: str, question: str) -> VQAInferenceResult:
    images, sizes, scales_yx = image_preprocess(image_path)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    features = output_dict.get("roi_features")
    question = [question]
    inputs = bert_tokenizer(
        question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    output_vqa = model(input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False)
    # print(f"Know answer output shape -> {output_vqa[4].shape}")
    
    # when using the original model
    # know_answer_pred = 1
    # answer_known = True
    # pred_vqa = output_vqa["logits"].argmax(-1)

    # when using my model
    know_answer_pred = output_vqa[4].argmax(-1)        
    pred_vqa = output_vqa[1].argmax(-1)

    answer = vqa_answers[pred_vqa]
    print(f"prediction from VisualBert VQA: {answer}, {know_answer_pred}" )

    answer_known = know_answer_pred.item() == 1

    return VQAInferenceResult(answer, answer_known)




VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"


vqa_answers = utils.get_data(VQA_URL)

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

image_preprocess = Preprocess(frcnn_cfg)
images_root_path = "../test-images"
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
vqa_model = VQAModel(visualbert_vqa.visual_bert, visualbert_vqa.cls, visualbert_vqa.config)
checkpoint = torch.load('../models/vqa2022-05-10_step0_epoch3.pth.tar')
vqa_model.load_state_dict(checkpoint['model_state_dict'])
vqa_model.eval()

app = Flask(__name__)
CORS(app)
app.config["DEBUG"]=True

@app.route('/', methods=['GET'])
def test_endpoint():
    return jsonify({'message':'Hello world'})


@app.route('/images-names', methods=['GET'])
def get_images_names():
    images_file_names = [f for f in os.listdir(images_root_path) if os.path.isfile(os.path.join(images_root_path, f))]
    #print("images file names are : {images_file_names}")
    return jsonify({"imagesFilenames":images_file_names})


@app.route('/vqa-inference', methods=['PUT'])
def vqa_inference():
    body = json.loads(request.data)
    question = body['question']
    img_name = body['imageName']
    print(f"Question: {question}, image name: {img_name}")
    print('Start perform inference')
    result = inference(vqa_model, image_preprocess, frcnn, bert_tokenizer, os.path.join('../test-images',img_name), question)
    print(f'Inference result: {result.answer}, {result.answer_known}')
    return {
        "answer": result.answer,
        "answerKnown":result.answer_known
        }


app.run()