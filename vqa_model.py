from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast, VisualBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
class VQAModel(nn.Module):
    def __init__(self, visual_bert: VisualBertModel, cls, config):
        super().__init__()

        self.config = config
        self.visual_bert = visual_bert
        self.cls = cls
        self.dropout = nn.Dropout(visual_bert.config.hidden_dropout_prob)
        
        visual_bert_hidden_dim = self.visual_bert.config.hidden_size
        # self.bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # VQA additional answer head
        self.answer_known_classifier = nn.Linear(visual_bert_hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the index of the last text token
        index_to_gather = attention_mask.sum(1) - 2  # as in original code

        # don't compute gradients for visual bert model
        with torch.no_grad():
            outputs = self.visual_bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
                image_text_alignment=image_text_alignment,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]

        # TO-CHECK: From the original code
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        )
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)

        pooled_output = self.dropout(pooled_output)

        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.config.num_labels)

        logits_know_answer = self.answer_known_classifier(pooled_output)
        reshaped_logits_know_answer = logits_know_answer.view(-1, 2)
        
        reshaped_logits_know_answer = self.sigmoid(reshaped_logits_know_answer)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (loss,reshaped_logits,outputs.hidden_states,outputs.attentions, reshaped_logits_know_answer)

