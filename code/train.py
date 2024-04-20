'''
Wenrui Liu
2024-4-20

train model
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

# parameter
max_epoch = 50
lr = 1e-5
L1Lambda = 100
# batch_size = 64

transformer_black = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
transformer_color = transforms.Compose([
    transforms.ToTensor(),
])

full_train_data = CIFARDataset("../data", "train", transformer_color, transformer_black)
train_size = int(0.9 * len(full_train_data))
validation_size = len(full_train_data) - train_size
train_data, validation_data = random_split(full_train_data, [train_size, validation_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size=256, shuffle=False)
# test_data = CIFARDataset("../data", "test", transformer_color, transformer_black)
# test_loader = DataLoader(test_data, batch_size=512, shuffle = False)

generator = Generator()
discriminator = Discriminator()
dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("GPU")
else:
    print("CPU")
generator = generator.to(dev)
discriminator = discriminator.to(dev)
bceLoss = nn.BCEWithLogitsLoss().to(dev)
L1Loss = nn.L1Loss().to(dev)
MSELoss = nn.MSELoss().to(dev)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=10*lr, betas=(0.5, 0.999))
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler_gen = StepLR(optimizer_gen, step_size=1, gamma=0.95)
scheduler_dis = StepLR(optimizer_dis, step_size=1, gamma=0.95)
optimizer_dis.zero_grad()
optimizer_gen.zero_grad()

dis_train_loss = [[], []]   # real and fake
gen_train_loss = [[], []]   # bce and l1
dis_val_loss = [[], []]
gen_val_loss = []


# train and validation
for epoch in range(max_epoch):
    dis_total_real_loss = 0
    dis_total_fake_loss = 0
    gen_bce_loss = 0
    gen_l1_loss = 0
    total = 0

    # train
    generator.train()
    discriminator.train()
    for batch_idx, (color_images, black_images, _) in enumerate(train_loader):
        total += color_images.size()[0]
        color_images, black_images = color_images.to(dev), black_images.to(dev)
        color_images_fake = generator(black_images)

        # discriminator
        optimizer_dis.zero_grad()
        result_real = discriminator(black_images, color_images)
        loss_real = bceLoss(result_real, torch.ones_like(result_real))

        result_fake = discriminator(black_images, color_images_fake.detach())
        loss_fake = bceLoss(result_fake, torch.zeros_like(result_fake))
        loss = (loss_real+loss_fake)/2

        loss.backward()
        optimizer_dis.step()

        dis_total_real_loss += loss_real.item()
        dis_total_fake_loss += loss_fake.item()

        optimizer_gen.zero_grad()
        # generator
        result_gen = discriminator(black_images, color_images_fake)
        loss_gen = bceLoss(result_gen, torch.ones_like(result_gen))
        loss_l1 = L1Loss(color_images_fake, color_images)*L1Lambda
        gen_bce_loss += loss_gen.item()
        gen_l1_loss += loss_l1.item()
        loss_gen = loss_gen + loss_l1

        loss_gen.backward()
        optimizer_gen.step()
        # if batch_idx%100 == 0:
        #     print("epoch: %d, batch: %d/%d, discriminator loss: %.4f, generator loss: %.4f" % (epoch, batch_idx, len(train_loader), loss.item(), loss_gen.item()))

        # gen_train_loss.append(loss_gen.item())
    dis_train_loss[0].append(dis_total_real_loss/total)
    dis_train_loss[1].append(dis_total_fake_loss/total)
    gen_train_loss[0].append(gen_bce_loss/total)
    gen_train_loss[1].append(gen_l1_loss/total)
    print("[epoch %d train] discriminator real loss: %.4f, discriminator fake loss: %.4f, generator BCE loss: %.4f, generator L1 loss: %.4f" %(epoch, dis_train_loss[0][-1], dis_train_loss[1][-1], gen_train_loss[0][-1], gen_train_loss[1][-1]))
    scheduler_dis.step()
    scheduler_gen.step()

    # validation
    generator.eval()
    discriminator.eval()
    dis_total_real_loss = 0
    dis_total_fake_loss = 0
    gen_total_loss = 0
    total = 0
    store_image = False
    for batch_idx, (color_images, black_images, image_idxs) in enumerate(validation_loader):
        total += color_images.size()[0]
        color_images, black_images = color_images.to(dev), black_images.to(dev)

        # discriminator
        color_images_fake = generator(black_images)

        # generator
        loss_gen = MSELoss(color_images_fake, color_images)
        gen_total_loss += loss_gen.item()
        # dis_val_loss[0].append(loss_real.item())
        # dis_val_loss[1].append(loss_fake.item())
        # gen_val_loss.append(loss_gen.item())

        if epoch%5 == 0 and not store_image:
            store_image = True
            to_pil = ToPILImage()
            # to_pil(color_images[0]).save("../result/epoch_%d_real_%d.png" % (epoch, image_idxs[0]))
            to_pil(color_images_fake[0]).save("../result/trainResult/epoch_%d_fake_%d.png" % (epoch, image_idxs[0]))
            # to_pil(color_images[1]).save("../result/epoch_%d_real_%d.png" % (epoch, image_idxs[1]))
            to_pil(color_images_fake[1]).save("../result/trainResult/epoch_%d_fake_%d.png" % (epoch, image_idxs[1]))
            # to_pil(color_images[2]).save("../result/epoch_%d_real_%d.png" % (epoch, image_idxs[2]))
            to_pil(color_images_fake[2]).save("../result/trainResult/epoch_%d_fake_%d.png" % (epoch, image_idxs[2]))
            # to_pil(color_images[3]).save("../result/epoch_%d_real_%d.png" % (epoch, image_idxs[3]))
            to_pil(color_images_fake[3]).save("../result/trainResult/epoch_%d_fake_%d.png" % (epoch, image_idxs[3]))
            # to_pil(color_images[4]).save("../result/epoch_%d_real_%d.png" % (epoch, image_idxs[4]))
            to_pil(color_images_fake[4]).save("../result/trainResult/epoch_%d_fake_%d.png" % (epoch, image_idxs[4]))

    gen_val_loss.append(gen_total_loss/total)
    print("[epoch %d validation] generator total MSE loss: %.4f" %(epoch, gen_val_loss[-1]*total))

# save 
torch.save(discriminator.state_dict(), "../result/discriminator.pth")
torch.save(generator.state_dict(), "../result/generator.pth")

#
# print(dis_train_loss[0], dis_train_loss[1], gen_train_loss)
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(dis_train_loss[0], label="discriminator real")
plt.plot(dis_train_loss[1], label="discriminator fake")
plt.title('Discriminator Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# print(dis_val_loss[0], dis_val_loss[1], gen_val_loss)
# plt.subplot(2, 2, 2)
# plt.plot(dis_val_loss[0], label="discriminator real")
# plt.plot(dis_val_loss[1], label="discriminator fake")
# plt.title('Discriminator Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

plt.subplot(2, 2, 2)
plt.plot(gen_train_loss[0], label="generator loss")
plt.title('Generator Training BCE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 3)
plt.plot(gen_train_loss[1], label="generator loss")
plt.title('Generator Training L1 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(gen_val_loss, label="generator loss")
plt.title('Generator Validation MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')


plt.show()