import json
import pickle
import numpy as np
from math import log, e
def entropy(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent
def compute_confidence_score(confidence_list):
    confidence_increment = {'yes': 1, 'maybe': 0.6, 'no': 0.1}
    score = [confidence_increment[conf] for conf in confidence_list]
    return np.array(score).sum()
    
def compute_qa_entropy(answers):
    qa_entropies = []
    for answer_entry in answers:
        answers = answer_entry['answers']
        answers_list = [a['answer'] for a in answers]
        confidence_list = [a['answer_confidence'] for a in answers]
        ent = entropy(answers_list)
        confidence_score = compute_confidence_score(confidence_list)
        qa_entropies.append({
            'question_id': answer_entry['question_id'],
            'answers_entropy': ent,
            'confidence_score': confidence_score
        })
    return qa_entropies
    
if __name__ == '__main__':
    # compute target for training dataset
    # train_answer_file = '../data/Annotations/v2_mscoco_train2014_annotations.json'
    # train_answers = json.load(open(train_answer_file))['annotations']

    val_answer_file = '../data/Annotations/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']

    #entropy_train = compute_qa_entropy(train_answers)
    #pickle.dump(entropy_train, open('../data/cache/train_qa_entropy.pkl','wb'))
    entropy_val = compute_qa_entropy(val_answers)
    pickle.dump(entropy_val, open('../data/cache/val_qa_entropy.pkl','wb'))

