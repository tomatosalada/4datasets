import torch
import torch.nn as nn

class CustomEEGNet(nn.Module):
    def __init__(self, numclasses=3):
        super(CustomEEGNet, self).__init__()
        
        # 入力: (Batch, Sequence, Freq, Channel) = (B, 16, 5, 18)
        # これを (B, 1, Channel*Freq, Sequence) = (B, 1, 90, 16) として扱います
        
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        
        # 1. Temporal Conv (時間方向の畳み込み)
        # kernel_size=(1, 5) -> 時間方向のみ5サンプル見る
        self.conv1 = nn.Conv2d(1, self.F1, (1, 5), padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(self.F1)
        
        # 2. Depthwise Conv (空間・周波数方向の畳み込み)
        # kernel_size=(90, 1) -> 縦方向(Freq*Ch)をまとめて1つの特徴にする
        self.conv2 = nn.Conv2d(self.F1, self.F1*self.D, (85, 1), groups=self.F1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.F1*self.D)
        self.act1 = nn.ELU()
        
        self.avgpool1 = nn.AvgPool2d((1, 2)) # 時間を半分に圧縮
        self.dropout1 = nn.Dropout(0.25)
        
        # 3. Separable Conv
        self.conv3 = nn.Conv2d(self.F1*self.D, self.F2, (1, 5), padding=(0, 2), groups=self.F1*self.D, bias=False)
        self.conv4 = nn.Conv2d(self.F2, self.F2, 1, bias=False) # Pointwise
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU()
        
        self.avgpool2 = nn.AvgPool2d((1, 2)) # さらに時間を半分に
        self.dropout2 = nn.Dropout(0.25)
        
        # 分類層
        # 時間16 -> pool(2) -> 8 -> pool(2) -> 4
        self.fc = nn.Linear(self.F2 * 4, numclasses)

    def forward(self, x):
        # x: (Batch, 16, 5, 18)
        
        # 1. 形を整える (Batch, 1, 90, 16)
        x = x.permute(0, 3, 2, 1) # -> (B, 18, 5, 16)
        x = x.reshape(x.shape[0], 1, -1, x.shape[3]) # -> (B, 1, 90, 16)
        
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
        #x = torch.sigmoid(self.fc(x))
        x = self.fc(x)
        return x

if __name__ == '__main__':
    input_data = torch.rand((32, 16, 5, 18))
    net = CustomEEGNet()
    output = net(input_data)
    print("Output shape:", output.shape)