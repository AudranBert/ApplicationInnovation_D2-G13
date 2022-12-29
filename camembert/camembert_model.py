# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# camembert = CamembertModel.from_pretrained("camembert-base")
import logging

# camembert.eval()  # disable dropout (or leave in train mode to finetune)


# Tokenize in sub-words with SentencePiece
# tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!']
# print(tokenized_sentence)
# 1-hot encode and add special starting and end tokens
# encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
# encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
# embeddings, _ = camembert(encoded_sentence, return_dict=False)
# embeddings = embeddings.detach().cpu()
# print(type(embeddings))
# print(embeddings.shape)
# print(embeddings.shape())
# embeddings.size torch.Size([1, 10, 768])
# tensor([[[-0.0254,  0.0235,  0.1027,  ..., -0.1459, -0.0205, -0.0116],
#         [ 0.0606, -0.1811, -0.0418,  ..., -0.1815,  0.0880, -0.0766],
#         [-0.1561, -0.1127,  0.2687,  ..., -0.0648,  0.0249,  0.0446],

from transformers import CamembertModel, CamembertTokenizer
import torch
import torch.nn as nn


device="cuda" if torch.cuda.is_available() else "cpu"
class CamembertCustom(nn.Module):

    def __init__(self):
        super(CamembertCustom, self).__init__()
        self.camembert = CamembertModel.from_pretrained("camembert-base").to(device)
        self.bidirectionnel = False
        self.hidden_dim = 32
        self.n_layers = 1
        self.dropout = 0
        self.gru = nn.GRU(512, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout,
                          bidirectional=self.bidirectionnel).to(device)
        self.fc = nn.Linear(self.hidden_dim * (self.bidirectionnel + 1), 10).to(device)
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers * (self.bidirectionnel + 1), batch_size, self.hidden_dim).zero_().to(device)
        return hidden


    def forward(self, x, h):
        # logging.info(x[0])
        # exit()
        # x = self.camembert(x)
        embeddings, _ = self.camembert(x, return_dict=False)
        embeddings = embeddings.detach()
        # logging.info(embeddings.shape)

        x = embeddings.transpose(1, 2)
        out, h = self.gru(x, h)
        a = nn.Sequential(
            self.relu,  # relu sur la sortie du gru
            self.fc  # puis couche lineaire
        )
        out = a(out)
        out = self.softmax(out)
        out = out[:, -1, :]
        return out, h

