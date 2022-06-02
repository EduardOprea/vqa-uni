from dataclasses import replace
from random import sample
from dataset.dataset import VQADataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
from vqa_model import VQAJustVisualFeaturesModel, VQAModel
import torch
import utils
import os
import h5py
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def save_checkpoint(path, epoch, model):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)

# **** INITIALIZING *****
h5_path = os.path.join("../data/preprocessed_img_features", 'train36.hdf5' )
hf = h5py.File(h5_path, 'r') 
dataset = VQADataset(hf)

torch.cuda.empty_cache()
device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device {device}")
know_answer_vqa = VQAJustVisualFeaturesModel().to(device)

# HYPERPARAMS, LOSS, OPTIMIZERS
n_epochs = 5
batch_size = 8
lr = 2e-4
b1 = .5
b2 = .999
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params = know_answer_vqa.answer_known_classifier.parameters(), lr = lr, betas=(b1,b2))

labels = [entry['answer']['label'] for entry in dataset.entries]
class_sample_count = np.array([len(np.where(labels == l)[0]) for l in np.unique(labels)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in labels])
samples_weight = torch.tensor(samples_weight)
imbalanced_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement = True)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=imbalanced_sampler, num_workers=0)

loss_list = []
step = 0
total_steps_epoch = int(dataset.__len__() / batch_size)
for epoch in range(n_epochs):
    step = 0
    print(f'Start epoch {epoch + 1 }/{n_epochs}')
    for index,(features,_,labels, _,_,_) in enumerate(train_loader):
        # clear gradients
        optimizer.zero_grad()

        output_vqa = know_answer_vqa(visual_embeds = features)
        # get prediction
        predicted_know_answer = output_vqa

        # compute loss between prediction and target
        target_one_hot = F.one_hot(labels, num_classes=2).to(torch.float32).to(device)

        loss = bce_loss(predicted_know_answer, target_one_hot)

        loss.backward()
        optimizer.step()

        if step % 25000 == 0:
            save_checkpoint(path=f'models/justvisual{epoch}_{step}.pth.tar',
             epoch= epoch,
             model=know_answer_vqa)
        print(f'Completed step {step}/{total_steps_epoch} with loss {loss}')
        
        step = step + 1


hf.close()