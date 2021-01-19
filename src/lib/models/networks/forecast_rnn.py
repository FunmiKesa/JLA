import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, device, input_size, num_hidden, num_layers):
        super(EncoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.encoder1 = nn.GRUCell(input_size, self.num_hidden)
        self.device = device
        self.num_layers = num_layers
        if num_layers > 1:
            self.encoder2 = nn.GRUCell(self.input_size, self.num_hidden)
        if num_layers > 2:
            self.encoder3 = nn.GRUCell(self.input_size, self.num_hidden)

    def forward(self, input, val=False):
        context = torch.tensor(torch.zeros(input.size(
            0), self.num_hidden, dtype=torch.float), device=self.device)

        forecast_sequence = input.size()[1]
        for i in range(forecast_sequence):
            inp = input[:, i, :]
            context = self.encoder1(inp, context)
            if self.num_layers > 1:
                context = self.encoder2(inp, context)
            if self.num_layers > 2:
                context = self.encoder3(inp, context)
        return context


class DecoderRNN(nn.Module):
    def __init__(self, device, input_size, output_size, num_hidden, dropout_p, num_layers):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.decoder1 = nn.GRUCell(int(self.num_hidden/2), self.num_hidden)
        self.decoder2 = nn.GRUCell(self.num_hidden//2, self.num_hidden)
        self.fc_in = nn.Linear(self.num_hidden, self.input_size)
        self.fc_out = nn.Linear(self.num_hidden, self.output_size)
        # self.dropout_context = nn.Dropout(p=dropout_p)
        # self.dropout_dla_features = nn.Dropout(p=dropout_p)
        self.relu_context = nn.ReLU()
        self.relu_output = nn.ReLU()
        # self.relu_dla_features = nn.ReLU()
        self.context_encoder = nn.Linear(
            self.num_hidden, int(self.num_hidden / 2))
        # self.dla_encoder = nn.Linear(256, int(self.num_hidden / 2))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context, dla_features=None, future_length=5,  past_length=10):
        outputs = []
        # h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        h_t = context
        # Fully connected
        encoded_context = self.context_encoder(context)
        # encoded_dla_features = self.dla_encoder(dla_features)
        # Dropout
        # encoded_context = self.dropout_context(encoded_context)
        # encoded_dla_features = self.dropout_dla_features(encoded_dla_features)
        # Relu
        encoded_context = self.relu_context(encoded_context)
        # encoded_dla_features = self.relu_dla_features(encoded_dla_features)
        result = []
        if self.training:
            # generate input
            decoded_inputs = []
            h_t = context

            for i in range(past_length-1, -1, -1):
                h_t = self.decoder1(encoded_context, h_t)
                input = self.fc_in(h_t)
                decoded_inputs += [input]
            decoded_inputs = torch.stack(decoded_inputs, 1)
            result.append(decoded_inputs)

        # encoded_context = torch.cat((encoded_context, dla_features), 1)

        h_t = context

        # forecast
        for i in range(future_length):
            h_t = self.decoder2(encoded_context, h_t)
            if self.num_layers > 1:
                h_t = self.decoder2(encoded_context, h_t)
            if self.num_layers > 2:
                h_t = self.decoder3(encoded_context, h_t)
            output = self.fc_out(self.relu_output(h_t))
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        # outputs = outputs.view(-1, future_length*self.output_size )
        result.append(outputs)

        return result


class Cumsum(nn.Module):
    def forward(self, last, pred):
        # cumsum
        cs = last + torch.cumsum(pred, dim=1)
        return cs


class ForeCastRNN(nn.Module):
    def __init__(self, device, input_size, output_size, future_length, num_hidden, num_layers,  dropout=0):
        super(ForeCastRNN, self).__init__()
        self.device = device
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.future_length = future_length
        self.dropout = dropout
        self.encoder = EncoderRNN(
            self.device, self.input_size, self.num_hidden, self.num_layers)
        self.decoder = DecoderRNN(
            self.device, input_size, output_size, self.num_hidden, self.dropout, self.num_layers)

        self.final = Cumsum()

    def forward(self, prev_bboxes, features=None):
        context = self.encoder(prev_bboxes)
        past_length = prev_bboxes.shape[1]

        output = self.decoder(
            context, features, self.future_length, past_length=past_length)

        last = prev_bboxes[:, -1:, :4]
        output[-1] = self.final(last, output[-1])

        return output
