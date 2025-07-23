import torch.utils.checkpoint
from torch import nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from src.attention_layer import *
import pickle


class Roberta_model(RobertaPreTrainedModel):
    def __init__(self, config, args, vocab):
        super().__init__(config)
        self.name = "plm_model"
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.attention = AttentionLayer(args=args, vocab=vocab)
        
        if "text_label" == args.attention_mode and args.is_trans:
            lab_file = args.label_titlefile
            with open(lab_file, 'rb') as file:
                labDescVec = pickle.load(file)
                print("labDescVec的key", labDescVec.keys())
                labDescVec = labDescVec['id2descVec']
            self.labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32).unsqueeze(0),
                                                 requires_grad=True)
        else:
            self.output_size = args.hidden_size * 2

            self.labDescVec=torch.nn.Parameter(torch.randn(vocab.label_num, self.output_size, dtype=torch.float32).unsqueeze(0),
                                           requires_grad=True)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            label_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()

        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        dynamic_embedding,logits_main = self.attention(x=hidden_output, label_batch=self.labDescVec)

        return {
            "dynamic_embedding": dynamic_embedding,
            "logits": logits_main
        }
