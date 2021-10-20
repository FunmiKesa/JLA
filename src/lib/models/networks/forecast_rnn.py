import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_hidden):
        super(EncoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.encoder1 = nn.GRUCell(input_size, self.num_hidden)

    def forward(self, input):
        context = torch.zeros(input.size(
            0), self.num_hidden, dtype=torch.float, device=input.device)

        forecast_sequence = input.size()[1]
        for i in range(forecast_sequence):
            inp = input[:, i, :]
            context = self.encoder1(inp, context)
        return context


class DecoderRNN(nn.Module):
    def __init__(self, input_size, output_size, num_hidden, use_embedding=False):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.output_size = output_size
        self.use_embedding = use_embedding
        self.decoder1 = nn.GRUCell(int(self.num_hidden/2), self.num_hidden)
        h = self.num_hidden if use_embedding else self.num_hidden // 2
        self.decoder2 = nn.GRUCell(h, self.num_hidden)
        self.fc_in = nn.Linear(self.num_hidden, self.input_size)
        self.fc_out = nn.Linear(self.num_hidden, self.output_size)
        self.relu_context = nn.ReLU()
        self.relu_output = nn.ReLU()
        # self.relu_dla_features = nn.ReLU()
        self.context_encoder = nn.Linear(
            self.num_hidden, int(self.num_hidden / 2))
        # self.dla_encoder = nn.Linear(self.num_hidden // 2, int(self.num_hidden / 2))

    def forward(self, context, dla_features=None, future_length=5,  past_length=10):
        outputs = []
        # Fully connected
        encoded_context = self.context_encoder(context)
        # Relu
        encoded_context = self.relu_context(encoded_context)
        result = []
        # if self.training:
        # generate input
        decoded_inputs = []
        h_t = context

        for i in range(past_length-1, -1, -1):
            h_t = self.decoder1(encoded_context, h_t)
            input = self.fc_in(h_t)
            decoded_inputs.insert(0, input)
            # decoded_inputs += [input]
        decoded_inputs = torch.stack(decoded_inputs, 1)
        result.append(decoded_inputs)

        if self.use_embedding:
            # encoded_dla_features = self.dla_encoder(dla_features)
            # encoded_dla_features = self.relu_dla_features(encoded_dla_features)
            encoded_context = torch.cat((encoded_context, dla_features), 1)

        h_t = context

        # forecast
        for i in range(future_length):
            h_t = self.decoder2(encoded_context, h_t)
            output = self.fc_out(self.relu_output(h_t))
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        result.append(outputs)

        return result


class Cumsum(nn.Module):
    def forward(self, last, pred):
        # cumsum
        cs = last + torch.cumsum(pred, dim=1)
        return cs


class ForeCastRNN(nn.Module):
    def __init__(self, input_size, output_size, future_length, num_hidden, use_embedding=False):
        super(ForeCastRNN, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.output_size = output_size
        self.future_length = future_length
        self.use_embedding = use_embedding
        self.encoder = EncoderRNN(self.input_size, self.num_hidden)
        self.decoder = DecoderRNN(
            input_size, output_size, self.num_hidden, self.use_embedding)

        self.final = Cumsum()

    def forward(self, prev_bboxes, features=None):
        context = self.encoder(prev_bboxes)
        past_length = prev_bboxes.shape[1]

        output = self.decoder(
            context, features, self.future_length, past_length=past_length)

        last = prev_bboxes[:, -1:, :4]
        output[-1] = self.final(last, output[-1])

        return output
