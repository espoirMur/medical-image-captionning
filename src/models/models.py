import torch
from torch import nn
from torch.nn.modules import dropout
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        self.fine_tune()
        
    def forward(self, images):
        with torch.no_grad():
            output = self.resnet(images)
        reset_features = output.reshape(output.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(reset_features))
        return final_features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.drop_prob = 0.2
        self.vocabulary_size = vocab_size
        self.lstm_layer = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embed_size)
        self.linear_layer = nn.Linear(hidden_size, self.vocabulary_size)
        self.embedding_layer.weight.data.uniform_(-0.1, 0.1)
        self.linear_layer.weight.data.uniform_(-0.1, 0.1)
        self.linear_layer.bias.data.fill_(0)

    def forward(self, captions, decoder_hidden):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(captions)  # shape s (batch_size, max_seq_len, embed_size)
        output, hidden = self.lstm_layer(embeddings, decoder_hidden)
        dropouted_outputs = self.dropout(output)
        model_outputs = self.linear_layer(dropouted_outputs) # shape (batch_size, max_seq_len, vocab_size)
        model_outputs = torch.permute(model_outputs, (0, 2, 1))  # permute to be able to compute cross entropy loss (batch_size, vocab_size, max_seq_len)
        return model_outputs

    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)          # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                        # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embed(predicted_outputs)                       # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices
