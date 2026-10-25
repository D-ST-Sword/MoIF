"""Fine-tuned RoBERTa-large with the original forward convention."""
from torch import nn
from transformers import RobertaModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained="roberta-large"):
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained)
        self.hidden_size = self.model.config.hidden_size
        if self.hidden_size != 1024:
            raise ValueError("The released MoIF configuration requires RoBERTa-large (1024 dimensions).")

    def forward(self, text):
        input_ids = text[:, 0, :].long()
        attention_mask = text[:, 1, :].float()
        input_ids = input_ids.clamp(min=0, max=self.model.config.vocab_size - 1)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
