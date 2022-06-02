from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
from dataset.dataset_no_visual import VQANoVisualFeaturesDataset
from vqa_model import VQAModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def save_checkpoint(path, epoch, model):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)


dataset = VQANoVisualFeaturesDataset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device {device}")
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
know_answer_vqa = VQAModel(visualbert_vqa.visual_bert, visualbert_vqa.cls, visualbert_vqa.config).to(device)

# HYPERPARAMS, LOSS, OPTIMIZERS
n_epochs = 5
batch_size = 16
lr = 2e-4
b1 = .5
b2 = .999
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params = know_answer_vqa.answer_known_classifier.parameters(), lr = lr, betas=(b1,b2))


# can not increase num_workers due to the inability to pickle h5files which is needed for forking workers subprocesses

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
    for index,(questions,labels, gt_answer, _, _) in enumerate(train_loader):
        # clear gradients
        optimizer.zero_grad()
        questions = list(questions)
        with torch.no_grad():  
            tokens = bert_tokenizer(
                questions,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

        # compute network prediction
        output_vqa = know_answer_vqa(input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
                token_type_ids=tokens.token_type_ids.to(device),
                output_attentions=False)
        # get prediction
        predicted_know_answer = output_vqa[4]
        # compute loss between prediction and target
        target_one_hot = F.one_hot(labels, num_classes=2).to(torch.float32).to(device)

        loss = bce_loss(predicted_know_answer, target_one_hot)

        loss.backward()
        optimizer.step()
        weights = know_answer_vqa.answer_known_classifier.weight
        print('Weights now:',weights)
        loss_list.append(loss)
        
        if step % 5000 == 0:
            save_checkpoint(path='testsave.pth.tar',
             epoch= epoch,
             model=know_answer_vqa)
        print(f'Completed step {step}/{total_steps_epoch} with loss {loss}')
        
        step = step + 1
