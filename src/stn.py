import torch
import torch.nn as nn
import torch.nn.functional as F

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation is very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class CNN_Vgg(nn.Module):
    def __init__(self, input_channel_num = 3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = input_channel_num, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),  # 32*64
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # 16*32
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # 8*16
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # 4*8
            nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), # 2*4
            nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # 1*2
            nn.BatchNorm2d(256), nn.ReLU(True),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, input):
        y = self.cnn(input)
        return y

class CNN_ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, input):
        y = self.cnn(input)
        return y

class TransformationConfig:
    def __init__(self, cnn = None):
        self.cnn_type = 'resnet'
#         self.cnn_type = 'vgg'
        self.input_channel_num = 3 # rgb
        self.control_points_size = (4,10) # shape of control points
        self.target_size = (32,100) # shape of output image 
        self.margins = (0.05, 0.05)
        self.num_weights = 4
        
        # the following are auto
        self.cnn = cnn
        if self.cnn_type=='vgg':
            self.outputsize = 256*2
        else:
#             self.outputsize = 256*16
            self.outputsize = 256*2*16

class Transformation(nn.Module):
    def __init__(self, config = TransformationConfig()):
        super().__init__()
        
        self.target_height, self.target_width = config.target_size
        self.num_lines, self.num_control_points = config.control_points_size
        self.num_weights = config.num_weights

        if config.cnn:
            self.cnn = config.cnn
        elif config.cnn_type=='vgg':
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels = input_channel_num, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),  # 32*64
                nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False), # 16*32
                nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False), # 8*16
                nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), # 4*8
                nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 128, 3, 1, 1, bias=False), # 2*4
                nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), # 1*2
                nn.BatchNorm2d(256), nn.ReLU(True),
            )
        else:
            self.cnn = ResNet(config.input_channel_num)
            
        self.fc1 = nn.Sequential(nn.Linear(config.outputsize, 128), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(128, self.num_control_points*self.num_lines+self.num_lines + self.num_weights)

        
        self.init_weights(self.cnn)
        self.init_weights(self.fc1)
        self.init_last_fc(self.fc2, config.margins)

        # output_ctrl_pts are specified, according to our task.
        margin_x, margin_y = config.margins
        ctrl_pts_x = torch.linspace(margin_x, 1.0 - margin_x, self.num_control_points)
        ctrl_pts_list=[]
        ctrl_pts_y_ = torch.linspace(margin_y, 1.0 - margin_y, self.num_lines)
        for i in range(self.num_lines):
            ctrl_pts_y = torch.ones(self.num_control_points) * ctrl_pts_y_[i]
            ctrl_pts = torch.stack([ctrl_pts_x, ctrl_pts_y], dim=1)
            ctrl_pts_list.append(ctrl_pts)
        target_control_points = torch.cat(ctrl_pts_list, dim=0)

        N = self.num_control_points*self.num_lines
        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        Y,X = torch.meshgrid(torch.linspace(0, 1,self.target_height), torch.linspace(0,1,self.target_width))
        X,Y = X.flatten(),Y.flatten()
        target_coordinate = torch.stack([X,Y], dim=1) # HW x 2

        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)
    
    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    def init_last_fc(self, fc, margins):
        marginx, marginy = margins
        ctrl_pts_x = torch.linspace(marginx, 1.-marginx, self.num_control_points)
        
        weight = torch.zeros(self.num_weights+self.num_lines)
        weight[:self.num_lines] = torch.linspace(marginy, 1.-marginy, self.num_lines)
        ctrl_weights = torch.cat([ctrl_pts_x]*self.num_lines+[weight],0)
        fc.weight.data.zero_()
        fc.bias.data = torch.Tensor(ctrl_weights)
       

    def forward(self, images):
        '''
        input [batch, 3, width, height]
        '''
        if images.shape[-1]!=64 or images.shape[-2]!=32:
            images = F.interpolate(images, (32, 64), mode='bilinear', align_corners=True)
        x = self.cnn(images)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
#         print(x.shape)
        img_feat = self.fc1(x) #img_feat.shape[256, 512]
        x = self.fc2(0.1 * img_feat)

        x = x.view(batch_size, -1)#x.shape[256, 30]
        #print(count)
        
        ctrl_pts_x, bias, weight = x.split((self.num_control_points*self.num_lines, self.num_lines, self.num_weights),1)
        #print("top",top.shape)#[256, 3\10]

        y_w = [ctrl_pts_x]
        for i in range(2, self.num_weights+1):
            y_w.append(ctrl_pts_x*y_w[-1])
        ctrl_pts_y_weight = torch.stack(y_w, 1) #ctrl_pts_y_weight.shape[256, 5, 10]
        #(batch, num_weights) * (num_weights, num_control_points) -> (batch, num_control_points)
        ctrl_pts_y_ = torch.bmm(weight.unsqueeze(1), ctrl_pts_y_weight).squeeze(1)
        ctrl_pts_y_ = ctrl_pts_y_.chunk(self.num_lines,1)
        ctrl_pts_x_ = ctrl_pts_x.chunk(self.num_lines,1)
        bias_ = bias.chunk(self.num_lines,1)
        ctrlpoints_=[]
        for x_, y_, b_ in zip(ctrl_pts_x_, ctrl_pts_y_, bias_):
          ctrlpoints_.append(torch.stack([x_, y_ + b_],dim=2))
        control_points = torch.cat(ctrlpoints_,dim=1)

        Y = torch.cat([control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        mapped_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)

        grid = mapped_coordinate.view(-1, self.target_height, self.target_width, 2)
        # the control_points may be out of [0, 1].
        grid = torch.clamp(grid, 0, 1)
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        images_rectified = F.grid_sample(images, grid)#[256, 3, 32, 100]

        return images_rectified, (control_points, bias, weight)