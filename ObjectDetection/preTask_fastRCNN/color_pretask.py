class color_encoder(nn.Module):
    def __init__(self):
        super(color_encoder, self).__init__()
        encoder = torchvision.models.resnet18(pretrained=False)
        encoder = list(encoder.children())[1:-3]
        encoder = [nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + encoder
        self.encoder = nn.Sequential(*encoder)
        
    def forward(self, x):
        x = self.encoder(x)
        return x 
    
class _SameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_layers, out_activation = 'relu'):
        super(_SameDecoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        ]*(num_layers-2)
        
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if out_activation == 'relu' else nn.Sigmoid(),
        ]
        self.decode = nn.Sequential(*layers)
    def forward(self, x):
        return self.decode(x)
    
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class color_decoder(nn.Module):
    def __init__(self):
        super(color_decoder, self).__init__()
        self.dec1 = _SameDecoder(256, 512,kernel_size=2, stride=2, num_layers=4)
        self.dec2 = _SameDecoder(512, 256,kernel_size=2, stride=2, num_layers=4)
        self.dec3 = _SameDecoder(256, 128,kernel_size=2, stride=2, num_layers=2)
        self.dec4 = _SameDecoder(128, 64,kernel_size=2, stride=2, num_layers=2)
        self.conv_out = nn.Conv2d(64, 2, 3 , padding=1)
        
        self.final_upsample = nn.Upsample((256, 306), mode='bilinear', align_corners=False)
        #self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                
    def forward(self, x):
        x = self.dec1(x)
        x= self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.conv_out(x)
        #print('after layers the size is', x.size())
        x = self.final_upsample(x)
        #x = self.sigmoid(x)
        return x
    
class color_model(nn.Module):
    def __init__(self):
        super(color_model, self).__init__()
        self.encoder = color_encoder()
        self.decoder = color_decoder()
        
    def forward(self, x):
        mid = self.encoder(x)
        output = self.decoder(mid)
        
        return output