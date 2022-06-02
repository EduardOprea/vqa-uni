from dataset.dataset import VQADataset
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
from vqa_model import VQAJustVisualFeaturesModel, VQAModel
import torch
import utils
import os
import h5py
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import pickle
import numpy as np

# **** INITIALIZING *****
name = 'val'
h5_path = os.path.join("../data/preprocessed_img_features", f'{name}36.hdf5' )
hf = h5py.File(h5_path, 'r') 
dataset = VQADataset(hf, name=name)

device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device {device}")
know_answer_vqa = VQAJustVisualFeaturesModel().to(device)
checkpoint = torch.load('models/justvisual4_50000.pth.tar')
know_answer_vqa.load_state_dict(checkpoint['model_state_dict'])



batch_size = 16

# extracting just a subset of the indices of the dataset
# n_entries_dataset = dataset.__len__()
# n_samples = int(5)
# arr = np.array([0] * (n_entries_dataset - n_samples) + [1]*n_samples)
# np.random.shuffle(arr)
# subset_indices = np.where(arr == 1)

# #subset_sampler = SubsetSampler(torch.tensor(arr))
# subset_sampler = SubsetRandomSampler(subset_indices)
# print(f"Using just {n_samples} samples from the dataset ")
# subset = Subset(dataset, subset_indices)


# can not increase num_workers due to the inability to pickle h5files which is needed for forking workers subprocesses
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

loss_list = []
total_steps_epoch = int(train_loader.dataset.__len__() / batch_size)
print(f"Total steps epoch {total_steps_epoch}")
n_batches_to_consider = 1000
step = 0

accuracy_metric = torchmetrics.Accuracy(num_classes=2, average='none')
f1_metric = torchmetrics.F1Score(num_classes=2, average='none')
precision_metric = torchmetrics.Precision(num_classes=2, average = "none")
recall_metric = torchmetrics.Recall(num_classes=2, average='none')
specificity_metric = torchmetrics.Specificity(num_classes=2, average = 'none')

know_answer_vqa.eval()

it = iter(train_loader)
# doing validation only on a small subset
with torch.no_grad():
    for i in range(n_batches_to_consider):
        features,questions,labels, gt_answers, _,_ = next(it)
        output_vqa = know_answer_vqa(visual_embeds = features)
        # get prediction
        predicted_know_answer = output_vqa
        f1_score = f1_metric(predicted_know_answer.cpu(), labels)
        precision_score = precision_metric(predicted_know_answer.cpu(), labels)
        recall_score = recall_metric(predicted_know_answer.cpu(), labels)
        accuracy_score = accuracy_metric(predicted_know_answer.cpu(), labels)
        specificity_score = specificity_metric(predicted_know_answer.cpu(), labels)

        print(f'Completed step {step}/{n_batches_to_consider}')
        step = step + 1
        

# with torch.no_grad():
#     for index,(features,questions,labels, gt_answers) in enumerate(train_loader):
#         questions = list(questions)
#         gt_answers = list(gt_answers)
#         concat_qa  = [q + ' ' + a for q,a in zip(questions,gt_answers)]
#         tokens = bert_tokenizer(
#                 questions,
#                 padding="max_length",
#                 max_length=20,
#                 truncation=True,
#                 return_token_type_ids=True,
#                 return_attention_mask=True,
#                 add_special_tokens=True,
#                 return_tensors="pt",
#             )

#         output_vqa = know_answer_vqa(input_ids=tokens.input_ids.to(device),
#                 attention_mask=tokens.attention_mask.to(device),
#                 visual_embeds=features.to(device),
#                 visual_attention_mask=torch.ones(features.shape[:-1]).to(device),
#                 token_type_ids=tokens.token_type_ids.to(device),
#                 output_attentions=False)
#         # get prediction
#         predicted_know_answer = output_vqa[4]
#         f1_score = f1_metric(predicted_know_answer.cpu(), labels)
#         precision_score = precision_metric(predicted_know_answer.cpu(), labels)
#         recall_score = recall_metric(predicted_know_answer.cpu(), labels)
#         accuracy_score = accuracy_metric(predicted_know_answer.cpu(), labels)
#         specificity_score = specificity_metric(predicted_know_answer.cpu(), labels)

#         print(f'Completed step {step}/{total_steps_epoch}')
#         step = step + 1

acc = accuracy_metric.compute()
precision = precision_metric.compute()
recall = recall_metric.compute()
f1 = f1_metric.compute()
specificity = specificity_metric.compute()

print(f"Accuracy on all data : {acc}")
print(f"Precision on all data : {precision}")
print(f"Recall on all data : {recall}")
print(f"F1 on all data : {f1}")
print(f"Specificity on all data : {specificity}")

performance = {
    "accuracy": acc.numpy(),
    "precision": precision.numpy(),
    "recall": recall.numpy(),
    "f1": f1.numpy(),
    "specificity": specificity.numpy()
}

pickle.dump(performance, open("performance.pkl", "wb"))


hf.close()