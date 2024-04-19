'''
Wenrui Liu
2024-4-16

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
from tqdm import tqdm

# parameter
max_epoch = 30
lr = 1e-4
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
validation_loader = DataLoader(validation_data, batch_size=256, shuffle=True)
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
bceLoss = nn.BCEWithLogitsLoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
scheduler_gen = StepLR(optimizer_gen, step_size=1, gamma=0.99)
scheduler_dis = StepLR(optimizer_dis, step_size=1, gamma=0.99)
optimizer_dis.zero_grad()
optimizer_gen.zero_grad()

dis_train_loss = [[], []]   # real and fake
gen_train_loss = []
dis_val_loss = [[], []]
gen_val_loss = []


# train and validation
for epoch in range(max_epoch):
    dis_total_real_loss = 0
    dis_total_fake_loss = 0
    gen_total_loss = 0
    total = 0

    # train
    generator.train()
    discriminator.train()
    for batch_idx, (color_images, black_images) in enumerate(train_loader):
        total += color_images.size()[0]
        color_images, black_images = color_images.to(dev), black_images.to(dev)

        # discriminator
        if batch_idx%5:
            optimizer_dis.zero_grad()
            color_images_fake = generator(black_images)
            result_real = discriminator(black_images, color_images)
            loss_real = bceLoss(result_real, torch.ones_like(result_real))
            result_fake = discriminator(black_images, color_images_fake.detach())
            loss_fake = bceLoss(result_fake, torch.zeros_like(result_fake))
            loss = (loss_real+loss_fake)/2

            loss.backward()
            optimizer_dis.step()

            dis_total_real_loss += loss_real.item()
            dis_total_fake_loss += loss_fake.item()
            dis_train_loss[0].append(loss_real.item())
            dis_train_loss[1].append(loss_fake.item())

        optimizer_gen.zero_grad()
        # generator
        color_images_fake = generator(black_images)
        result_gen = discriminator(black_images, color_images_fake)
        loss_gen = bceLoss(result_gen, torch.ones_like(result_gen))
        gen_total_loss += loss_gen.item()

        loss_gen.backward()
        optimizer_gen.step()
        # if batch_idx%100 == 0:
        #     print("epoch: %d, batch: %d/%d, discriminator loss: %.4f, generator loss: %.4f" % (epoch, batch_idx, len(train_loader), loss.item(), loss_gen.item()))

        gen_train_loss.append(loss_gen.item())
    # dis_train_loss[0].append(dis_total_real_loss/total)
    # dis_train_loss[1].append(dis_total_fake_loss/total)
    # gen_train_loss.append(gen_total_loss/total)
    print("[epoch %d train] discriminator real loss: %.4f, discriminator fake loss: %.4f, generator loss: %.4f" %(epoch, dis_train_loss[0][-1], dis_train_loss[1][-1], gen_train_loss[-1]))
    scheduler_dis.step()
    scheduler_gen.step()

    # validation
    generator.eval()
    discriminator.eval()
    dis_total_real_loss = 0
    dis_total_fake_loss = 0
    gen_total_loss = 0
    total = 0
    for batch_idx, (color_images, black_images) in enumerate(validation_loader):
        total += color_images.size()[0]
        color_images, black_images = color_images.to(dev), black_images.to(dev)

        # discriminator
        color_images_fake = generator(black_images)
        result_real = discriminator(black_images, color_images)
        loss_real = bceLoss(result_real, torch.ones_like(result_real))
        result_fake = discriminator(black_images, color_images_fake.detach())
        loss_fake = bceLoss(result_fake, torch.zeros_like(result_fake))
        loss = (loss_real+loss_fake)/2

        dis_total_real_loss += loss_real.item()
        dis_total_fake_loss += loss_fake.item()

        # generator
        result_gen = discriminator(black_images, color_images_fake)
        loss_gen = bceLoss(result_gen, torch.ones_like(result_gen))
        gen_total_loss += loss_gen.item()
        # dis_val_loss[0].append(loss_real.item())
        # dis_val_loss[1].append(loss_fake.item())
        # gen_val_loss.append(loss_gen.item())
    dis_val_loss[0].append(dis_total_real_loss/total)
    dis_val_loss[1].append(dis_total_fake_loss/total)
    gen_val_loss.append(gen_total_loss/total)
    print("[epoch %d validation] discriminator real loss: %.4f, discriminator fake loss: %.4f, generator loss: %.4f" %(epoch, dis_val_loss[0][-1], dis_val_loss[1][-1], gen_val_loss[-1]))

# save 
torch.save(discriminator.state_dict(), "../result/discriminator.pth")
torch.save(generator.state_dict(), "../result/generator.pth")

#
# print(dis_train_loss[0], dis_train_loss[1], gen_train_loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dis_train_loss[0], label="discriminator real")
plt.plot(dis_train_loss[1], label="discriminator fake")
plt.plot(gen_train_loss, label="generator loss")
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# print(dis_val_loss[0], dis_val_loss[1], gen_val_loss)
plt.subplot(1, 2, 2)
plt.plot(dis_val_loss[0], label="discriminator real")
plt.plot(dis_val_loss[1], label="discriminator fake")
plt.plot(gen_val_loss, label="generator loss")
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()