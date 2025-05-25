import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim)) 

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat += self.pos_embed[:, :H*W]  
        attn_output, _ = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(attn_output + x_flat)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)
        out += self.shortcut(identity)
        return out

class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.attention = SelfAttention(embed_dim=512, num_heads=8)

        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128), 
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),          
            ResidualBlock(256, 256, downsample=True),
            ResidualBlock(256, 512),            
            ResidualBlock(512, 512),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.2)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4672) 
        )
        self.value_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output