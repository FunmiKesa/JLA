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
        context = torch.zeros(input.size(1), self.num_hidden, dtype=torch.float).to(self.device)

        forecast_sequence = input.size()[0]
        for i in range(forecast_sequence):
            inp = input[i, :, :]
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
        self.decoder1 = nn.GRUCell(int(self.num_hidden/2), self.num_hidden)
        self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        self.fc_in = nn.Linear(self.num_hidden, self.input_size)
        self.fc_out = nn.Linear(self.num_hidden, self.output_size)
        self.dropout_context = nn.Dropout(p=dropout_p)
        self.dropout_dla_features = nn.Dropout(p=dropout_p)
        self.relu_context = nn.ReLU()
        self.relu_dla_features = nn.ReLU()
        self.device = device
        self.context_encoder = nn.Linear(self.num_hidden, int(self.num_hidden / 2))
        self.dla_encoder = nn.Linear(256, int(self.num_hidden / 2))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context, dla_features, forecast_length=5,  sequence_length=10, val=False):
        outputs = []
        # h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        h_t = context
        # Fully connected
        encoded_context = self.context_encoder(context)
        # encoded_dla_features = self.dla_encoder(dla_features)
        # Dropout
        encoded_context = self.dropout_context(encoded_context)
        # encoded_dla_features = self.dropout_dla_features(encoded_dla_features)
        # Relu
        encoded_context = self.relu_context(encoded_context)
        # encoded_dla_features = self.relu_dla_features(encoded_dla_features)
        result = []
        if not val:
            # generate input
            decoded_inputs = []
            h_t = context
            
            for i in range(sequence_length-1, -1, -1):
                h_t = self.decoder1(encoded_context, h_t)
                input = self.fc_in(h_t)
                decoded_inputs += [input]
            decoded_inputs = torch.stack(decoded_inputs, 0)
            result.append(decoded_inputs)

        encoded_context = torch.cat((encoded_context, dla_features), 1)
        
        h_t = context

        # forecast
        for i in range(forecast_length):
            h_t = self.decoder2(encoded_context, h_t)
            if self.num_layers > 1:
                h_t = self.decoder2(encoded_context, h_t)
            if self.num_layers > 2:
                h_t = self.decoder3(encoded_context, h_t)
            output = self.fc_out(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        outputs = outputs.view(-1, forecast_length*self.output_size )
        result.append(outputs)
        
        return result

class ForeCastRNN(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(ForeCastRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRUCell(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.hidden_size//2)

        self.rnn_decode = nn.GRUCell(self.hidden_size//2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, input_size)

    def forward(self, input, prev_state=None):
        s, b, input_size = input.shape
        hx, cx = prev_state
            
        hidden_states = []
        for i in range(s):
            hx, cx = self.rnn(input[i], (hx, cx))
            hidden_states.append((hx, cx))
        hx = self.relu(hx)
        zf = self.fc(hx)
        # zf = self.relu(zf)

        decoded_output = []
        
        for i in range(s-1,-1, -1):
            hx, cx = self.rnn_decode(zf, hidden_states[i])
            # hx = self.relu(hx)
            fc_out = self.fc2(hx)
            decoded_output.insert(0, fc_out)

        if len(decoded_output) > 0:
            decoded_output = torch.stack(decoded_output, 0)
        
        # print(hidden_states[0], decoded_output[0])
        return hidden_states, decoded_output