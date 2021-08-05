import os 

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

# VGG19에서 conv layer만 가져옴
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.feature_maps = [0, 5, 10, 19, 28]
        self.model = models.vgg19(pretrained = True).features[:29]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)

            if i in self.feature_maps:
                features.append(x)

        return features

def load_image(img, transform):
    image = Image.open(img)
    image = transform(image)
    return image.unsqueeze(0)

data_list = os.listdir('./dataset')
style_path = list(filter(lambda x : 'style' in x, data_list))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

original_img = load_image('./dataset/무민이.jpeg', transform).to(device)

style_list = []
for path in style_path:
    path = os.path.join('./dataset', path)
    style_list.append(load_image(path, transform))


# output = torch.randn(size = original_img.data.shape, requires_grad=True).to(device)
output = original_img.clone().requires_grad_()
model = VGG().to(device)

epochs = 1000
lr = 1e-2
alpha = 1
beta = 1e4
optimizer = optim.Adam([output], lr = lr)
L2loss = nn.MSELoss()

for i, style_img in enumerate(style_list):
    print(f'Training is starting for style{i+1}')
    style_img = style_img.to(device)
    for epoch in range(epochs):
        
        origin_features = model(original_img)
        style_features = model(style_img)
        output_features = model(output)
        
        loss_content = 0
        loss_style = 0
        
        for origin, style, out in zip(origin_features, style_features, output_features):
    
            batch_size, channel, width, height = origin.shape
            # Content Loss
            # output과 original image의 형태를 유지
            # (origin_feature - noise_feature) ** 2
            loss_content += L2loss(origin, out)

            # Style Loss
            # output과 style image의 style의 상관성이 유사하게끔 학습
            # Gram matrix
            # torch.mm
            # [M, N] x [N, M] = [M, M]
            Gram_output = torch.mm(out.view(channel, -1), out.view(channel, -1).T)
            Gram_style = torch.mm(style.view(channel, -1), style.view(channel, -1).T)

            loss_style += L2loss(Gram_output, Gram_style)
        
        loss = alpha * loss_content + beta * loss_style

        # backward
        optimizer.zero_grad()
        loss_style.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'[{epoch + 1}/{epochs}] | total loss : {loss.item()} | loss content : {loss_content.item():.4f} | loss style : {loss_style.item()}')

    if not os.path.isdir('./output'):
        os.makedirs('./output')
    save_image(output, f'./output/style{i + 1}_output.png')