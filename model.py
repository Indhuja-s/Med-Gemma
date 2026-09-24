import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.down1 = double_conv(1, 64)
        self.pool = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(128, 64) # 128 because of skip connection
        
        self.out = nn.Conv2d(64, 1, kernel_size=1) # Output 1 channel (Tumor vs No Tumor)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        
        # Decoder
        t1 = self.up1(c2)
        # Skip connection: attach c1 to t1
        combined = torch.cat([t1, c1], dim=1)
        c3 = self.conv_up1(combined)
        
        return torch.sigmoid(self.out(c3))

# Test the model
if __name__ == "__main__":
    model = SimpleUNet()
    dummy_input = torch.randn(1, 1, 128, 128) # (Batch, Channel, Height, Width)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")