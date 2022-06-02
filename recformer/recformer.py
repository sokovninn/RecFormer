import torch
from torch import nn

class RecFormer(nn.Module):
  def __init__(self, num_tokens, num_labels, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, use_pretrained_embeddings = False):
    super(RecFormer, self).__init__()
    self.use_pretrained_embeddings = use_pretrained_embeddings
    if not use_pretrained_embeddings:
      self.embedding = nn.Embedding(num_tokens, dim_model)
    self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
    self.linear = nn.Linear(dim_model, num_labels)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, input_tensor):
    if not self.use_pretrained_embeddings:
      x = self.embedding(input_tensor).squeeze(1)
      x = self.dropout(x)
    else:
      x = input_tensor
    transformer_out = self.transformer(x, x)
    transformer_out = self.dropout(transformer_out)
    out = self.linear(torch.mean(transformer_out, dim=1))
    return out


class MiltitaskRecFormer(nn.Module):
  def __init__(self, num_tokens, num_labels, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, use_pretrained_embeddings = False):
    super(MiltitaskRecFormer, self).__init__()
    self.use_pretrained_embeddings = use_pretrained_embeddings
    if not use_pretrained_embeddings:
      self.embedding = nn.Embedding(num_tokens, dim_model)
    self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
    self.linear_cuisine = nn.Linear(dim_model, num_labels)
    self.linear_ingredients = nn.Linear(dim_model, num_tokens)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, input_tensor):
    if not self.use_pretrained_embeddings:
      x = self.embedding(input_tensor).squeeze(1)
      x = self.dropout(x)
    else:
      x = input_tensor
    transformer_out = self.transformer(x, x)
    transformer_out = self.dropout(transformer_out)
    out_cuisine = self.linear_cuisine(torch.mean(transformer_out, dim=1))
    out_ingredients = self.linear_ingredients(torch.mean(transformer_out, dim=1))
    return out_cuisine, out_ingredients