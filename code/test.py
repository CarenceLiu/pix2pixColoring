'''
Wenrui Liu
2024-4-20

test model
'''
from model import Generator, Discriminator
from CIFARDataset import CIFARDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torch.nn import init

transformer_black = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
transformer_color = transforms.Compose([
    transforms.ToTensor(),
])

test_data = CIFARDataset("../data", "test", transformer_color, transformer_black)
test_loader = DataLoader(test_data, batch_size=256, shuffle = False)

generator = Generator()
generator.load_state_dict(torch.load('../result/generator.pth'))
discriminator = Discriminator()
discriminator.load_state_dict(torch.load('../result/discriminator.pth'))
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
generator = generator.to(dev)
discriminator = discriminator.to(dev)

gen_total_loss = 0
total = 0
MSELoss = nn.MSELoss().to(dev)
generator.eval()
discriminator.eval()
for batch_idx, (color_images, black_images, image_idxs) in enumerate(test_loader):
    total += color_images.size()[0]
    color_images, black_images = color_images.to(dev), black_images.to(dev)
    color_images_fake = generator(black_images)
    loss_gen = MSELoss(color_images_fake, color_images)
    gen_total_loss += loss_gen.item()

    to_pil = ToPILImage()
    to_pil(color_images_fake[0]).save("../result/testResult/fake_%d.png" % (image_idxs[0]))
print("[test] generator MSE loss: %.4f" %(gen_total_loss/total))