import scipy.stats as stats
import pickle



if __name__ == '__main__':
   entropy_train = pickle.load(open('../../data/cache/train_qa_entropy.pkl','rb'))
   entropy_train = sorted(entropy_train, key=lambda x: x['question_id'])
   knowanswer_labels = sorted(pickle.load(open('../../data/cache/knowanswer_train_target.pkl','rb')),key=lambda x: x['question_id'])
   entropies = [entry['answers_entropy'] for entry in entropy_train]
   confidence_scores = [entry['confidence_score'] for entry in entropy_train]
   labels = [entry['label'] for entry in knowanswer_labels]
   print(len(labels))
   result_labels_entropies = stats.pointbiserialr(labels, entropies)
   print(f'The correlation between the labels and the entropies is {result_labels_entropies.correlation} \
   with p value {result_labels_entropies.pvalue}')

   result_labels_confidence = stats.pointbiserialr(labels, confidence_scores)
   print(f'The correlation between the labels and answers confidence is {result_labels_confidence.correlation} \
   with p value {result_labels_confidence.pvalue}')
   print(f'The pearson correlation coefficient between answers entropy and confidence is : {stats.pearsonr(entropies, confidence_scores)}')
