import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        ----------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)      # 所有时间点的隐层状态
            last_state_list.append([h, c])              # 最后一个时间点的隐层状态以及cell状态

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class inputVGG(nn.Module):
    def __init__(self, in_channels, out_channels, block_nums=None):
        super(inputVGG, self).__init__()
        if block_nums is None:
            block_nums = [1, 1, 1, 3, 3]
        self.stage1 = self._make_layers(in_channels=in_channels, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        # self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        # self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        self.stage5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        # self.fc1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(11,20), bias=False)
        # self.fc2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), bias=False)
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(256)
        # self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=(11,20), bias=False)
        # self.unstage5 = self._Demake_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        # self.unstage4 = self._Demake_layers(in_channels=512, out_channels=256, block_num=block_nums[3])
        self.unstage3 = self._Demake_layers(in_channels=256, out_channels=128, block_num=block_nums[2])
        self.unstage2 = self._Demake_layers(in_channels=128, out_channels=64, block_num=block_nums[1])
        self.unstage1 = self._Demake_layers(in_channels=64, out_channels=out_channels, block_num=block_nums[0])
        self.unpool = nn.MaxUnpool2d(2, 2)
        # self.unpool5 = nn.MaxUnpool2d(kernel_size=(2,2), stride=(3,2))
        self.unpool4 = nn.MaxUnpool2d(kernel_size=(3, 2), stride=(2, 2))
        # self.unpool3 = nn.MaxUnpool2d(kernel_size=(2,2), stride=(2,2))
        # self.unpool2 = nn.MaxUnpool2d(kernel_size=(3,2), stride=(2,2))
        # self.unpool1 = nn.MaxUnpool2d(kernel_size=(2,2), stride=(2,2))
        # self._init_params()

    def forward(self, x):
        x, u1 = self.stage1(x)
        x, u2 = self.stage2(x)
        x, u3 = self.stage3(x)
        x, u4 = self.stage4(x)
        x, u5 = self.stage5(x)
        # x = self.relu(self.bn(self.fc1(x)))
        # x = self.relu(self.bn(self.fc2(x)))
        # x = self.relu(self.bn(self.deconv(x)))
        x = self.unpool(x, u5)
        # x = self.unstage5(x)
        x = self.unpool4(x, u4)
        # x = self.unstage4(x)
        x = self.unpool(x, u3)
        x = self.unstage3(x)
        # x = self.unpool(x, u2)
        x = self.unstage2(x)
        # x = self.unpool(x, u1)
        x = self.unstage1(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    @staticmethod
    def _Conv3x3BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _DeConv3x3BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = [self._Conv3x3BNReLU(in_channels, out_channels)]
        for i in range(1, block_num):
            layers.append(self._Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), return_indices=True))
        return nn.Sequential(*layers)

    def _Demake_layers(self, in_channels, out_channels, block_num):
        layers = [self._DeConv3x3BNReLU(in_channels, out_channels)]
        for i in range(1, block_num):
            layers.append(self._DeConv3x3BNReLU(out_channels, out_channels))
        return nn.Sequential(*layers)


class outputCNN(nn.Module):
    def __init__(self, input_dim):
        super(outputCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=(2, 2))

    def forward(self,x):
        x = F.relu(self.conv1(x))
        output_size = x.shape
        x, i = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.unpool(x, i, output_size=output_size)
        x = F.relu(self.conv3(x))
        # x = torch.sigmoid(self.conv4(x))
        x = torch.relu(self.conv4(x))
        return x


class ConvLSTM_model(nn.Module):
    def __init__(self, VGG_indim, VGG_outdim, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, block_nums=None):
        super(ConvLSTM_model, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # self.inputVGG = inputVGG(in_channels=VGG_indim, out_channels=VGG_outdim, block_nums=block_nums)
        self.conv_lstm = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers,
                                  self.batch_first, self.bias, self.return_all_layers)
        self.outputCNN = outputCNN(self.hidden_dim)

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
            output_list 是最后一层的所有时间点状态的叠加之后的结果
            output 是 output_list 最后一个时间点 [隐层状态, cell状态]
        """
        output_list, output = self.conv_lstm(x)

        conv_output_list = []
        for i in range(output_list[0].size(1)):
            conv_output_list.append(self.outputCNN(output_list[0][:, i, :, :, :]))
        # 是不是没必要使用这么复杂的网络
        conv_output = [self.outputCNN(output[0][0]), self.outputCNN(output[0][1])]
        conv_output_list_ret = torch.stack(conv_output_list, dim=1)

        # conv_output_list_ret = output_list[0]
        # conv_output = output[0]

        return output_list, output, conv_output, conv_output_list_ret
