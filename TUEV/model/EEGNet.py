import torch
import torch.nn as nn

class CustomEEGNet(nn.Module):
    def __init__(self, numclasses=6, seq_len=10, freq=5, channel=16):
        super(CustomEEGNet, self).__init__()
        
        # 入力: (Batch, Sequence, Freq, Channel)
        # これを (B, 1, Channel*Freq, Sequence) として扱います
        
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        
        self.seq_len = seq_len
        self.freq = freq
        self.channel = channel
        self.spatial_dim = freq * channel
        
        # 1. Temporal Conv (時間方向の畳み込み)
        self.conv1 = nn.Conv2d(1, self.F1, (1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.F1)
        
        # 2. Depthwise Conv (空間・周波数方向の畳み込み)
        self.conv2 = nn.Conv2d(self.F1, self.F1*self.D, (self.spatial_dim, 1), groups=self.F1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.F1*self.D)
        self.act1 = nn.ELU()
        
        self.avgpool1 = nn.AvgPool2d((1, 2)) # 時間を半分に圧縮
        self.dropout1 = nn.Dropout(0.25)
        
        # 3. Separable Conv
        self.conv3 = nn.Conv2d(self.F1*self.D, self.F2, (1, 3), padding=(0, 1), groups=self.F1*self.D, bias=False)
        self.conv4 = nn.Conv2d(self.F2, self.F2, 1, bias=False) # Pointwise
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU()
        
        self.avgpool2 = nn.AvgPool2d((1, 2)) # さらに時間を半分に
        self.dropout2 = nn.Dropout(0.25)
        
        # 分類層の入力サイズ計算
        out_seq = self.seq_len // 2
        out_seq = out_seq // 2
        self.fc = nn.Linear(self.F2 * out_seq, numclasses)

    def forward(self, x):
        # x: (Batch, Sequence, Freq, Channel)
        
        # 1. 形を整える (Batch, 1, Channel*Freq, Sequence)
        x = x.permute(0, 3, 2, 1) # -> (B, Channel, Freq, Sequence)
        x = x.reshape(x.shape[0], 1, -1, x.shape[3]) # -> (B, 1, Channel*Freq, Sequence)
        
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # テスト用のダミーデータ: (Batch, Sequence, Freq, Channel) = (32, 10, 5, 16)
    input_data = torch.rand((32, 10, 5, 16))
    net = CustomEEGNet()
    output = net(input_data)
    print("Output shape:", output.shape)