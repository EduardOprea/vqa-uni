from collections import Counter
import os
import pickle

import torch

import torchmetrics


# targets_path = os.path.join("../data/cache", f'knowanswer_train_target.pkl')
# targets = pickle.load(open(targets_path, 'rb'))

# train_labels = [entry['label'] for entry in targets]

# print(Counter(train_labels))
accuracy_metric = torchmetrics.Accuracy(num_classes=2, average='none')
f1_metric = torchmetrics.F1Score(num_classes=2, average='none')
precision_metric = torchmetrics.Precision(num_classes=2, average = "none")
recall_metric = torchmetrics.Recall(num_classes=2, average='none')
specificity_metric = torchmetrics.Specificity(num_classes=2, average = 'none')
confmat_metric = torchmetrics.ConfusionMatrix(num_classes=2)
# gt = torch.tensor([1,0, 1,0])
# # TP - 2, FP - 1, TN - 0, FN - 0
# # recall should be 1
# pred = torch.tensor([[0.1,0.9],[0.1,0.9],[0.1,0.9], [0.9,0.1]])

# print("Accuracy:", accuracy_metric(pred,gt))
# print("Precision :", precision_metric(pred,gt))
# print("Recall:", recall_metric(pred,gt))
# print("Specificity", specificity_metric(pred,gt))

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 2).softmax(dim=-1)
    target = torch.randint(2, (10,))

    #print(preds)
    #print(target)
    # metric on current batch
    acc = accuracy_metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")
    print(f"F1 on batch {i}: {f1_metric(preds,target)}")
    print(f"Precision on batch {i}: {precision_metric(preds,target)}")
    print(f"Recall on batch {i}: {recall_metric(preds,target)}")
    print(f"Confusion on batch {i}: {confmat_metric(preds,target)}")
   


# metric on all batches using custom accumulation
acc = accuracy_metric.compute()
print(f"Accuracy on all data: {acc}")
print(f"F1 on all data: {f1_metric.compute()}")
print(f"Precision on all data: {precision_metric.compute()}")
print(f"Recall on all data: {recall_metric.compute()}")
print(f'Confusion matrix:', confmat_metric.compute())