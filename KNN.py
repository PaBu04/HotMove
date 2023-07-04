'''
Created on 06.05.2021

@author: Paul Buda
'''

import torch
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import win32api

import pygame
import numpy as np
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])

train_data_list = []
target_list = []
train_data = []
files = listdir("new/trainData")
for i in range(len(listdir("new/trainData"))):
    r = random.choice(files)
    files.remove(r)
    img = Image.open("new/trainData/" + r)
    img_tensor = transform(img)
    train_data_list.append(img_tensor)
    isCircle = 1 if 'circle' in r else 0
    isTriangle = 1 if 'triangle' in r else 0
    isLine = 1 if 'line' in r else 0
    isRandom = 1 if 'random' in r else 0
    target = [isCircle, isTriangle, isLine, isRandom]
    target_list.append(target)
    if(len(train_data_list) >= 64):
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, kernel_size = 3)
        self.conv2 = nn.Conv2d(7, 11, kernel_size = 3)
        self.conv3 = nn.Conv2d(11, 15, kernel_size = 3)
        
        self.fc1 = nn.Linear(540, 130)
        self.fc2 = nn.Linear(130, 4)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.size())
        x = x.view(-1, 540)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

#if(os.path.isfile('meinNetz.pt')):
    #model = torch.load('meinNetz.pt')
#else:
model = Netz()
model = model.cuda()

loss_list = []
optimizer = optim.Adam(model.parameters(), lr = 0.001)
def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #    epoch, (batch_id + 1) * len(data), len(train_data) * len(data), 100 * batch_id / len(train_data), loss.item()), "Output:", out[0], "Target:", target[0])
        batch_id += 1
    if(epoch % 10 == 0):
        loss_list.append(loss.item())
     
test_accuracy = []
def test():
    correct = 0
    for r in range(50):
        model.eval()
        files = listdir("new/testData")
        f = random.choice(files)
        img = Image.open("new/testData/" + f)
        img_evel_tensor = transform(img)
        img_evel_tensor.unsqueeze_(0)
        data = Variable(img_evel_tensor.cuda())
        out = model(data)
        if('circle' in f):
            target = 0
        elif('triangle' in f):
            target = 1
        elif('line' in f):
            target = 2
        else:
            target = 3
        #print("It's:", out.data.max(1, keepdim = True)[1].item())
        if(out.data.max(1, keepdim = True)[1].item() == target):
            correct += 1
        #img.show()
        #input("Next:")
    test_accuracy.append(correct * 2)
        

def getLabel(lastMoves):
    model.eval()
    img = Image.open("liveTest/move.jpeg")
    img_evel_tensor = transform(img)
    img_evel_tensor.unsqueeze_(0)
    data = Variable(img_evel_tensor.cuda())
    out = model(data)
    #print(out[0][0].item(), ",", out[0][1].item(), ",", out[0][2].item(), ",", out.data.max().item())
    print("It's:", out.data.max(1, keepdim = True)[1].item())
    
    lastMoves = np.append(lastMoves, out.data.max(1, keepdim = True)[1].item())
    lastMoves = np.delete(lastMoves, [0])
    if(lastMoves[0] == lastMoves[1] == lastMoves[2]):
        if(lastMoves[0] == 0):
            print("Circle detected!")
        elif(lastMoves[0] == 1):
            print("Triangle detected!")
        elif(lastMoves[0] == 2):
            print("Line detected!")
        lastMoves = np.array([4, 4, 4])
    return lastMoves
    
def liveTest():
    lastMoves = np.array([4, 4, 4])
    running = True

    #Zaehlt alle je gespeicherte Pixel
    number = 0
    #Laenge der sichtbaren Pixelkette
    length = 250
    
    #Fenster initialisieren
    screen = pygame.display.set_mode((1000, 1000))
    screen.fill("white")
    pygame.display.update()
    
    #Liste mit Koordinaten der aktiven Pixel
    dots = np.ones(length * 2)
    
    currentpos = win32api.GetCursorPos()
    while running:
        currentpos = win32api.GetCursorPos()
        if(pygame.event.poll().type == pygame.QUIT):
            running = False
        if(currentpos != win32api.GetCursorPos()):
            number += 1
            currentpos = win32api.GetCursorPos()
            
            
            #Neues aktives Pixel hinzufuegen
            dots = np.append(dots, [currentpos[0], currentpos[1]])
            #pygame.draw.circle(screen, "black", [currentpos[0], currentpos[1]], 2)
            
            #Letzten der aktiven Pixel entfernen
            pygame.draw.circle(screen, "white", (int(dots[0]), int(dots[1])), 2)
            dots = np.delete(dots, [0, 1])
            
            #pygame.display.update()
            
            if(number % 200 == 0): #macht alle 100 Pixel ein Screenshot
                #move Image in the upper left corner
                dots = dots.reshape(250, 2)
                lowestX = dots[0, 0]
                lowestY = dots[0, 1]
                highestX = dots[0, 0]
                highestY = dots[0, 1]
                for r in dots[1:]:
                    if(r[0] < lowestX):
                        lowestX = r[0]
                    if(r[0] > highestX):
                        highestX = r[0]
                    if(r[1] < lowestY):
                        lowestY = r[1]
                    if(r[1] > highestY):
                        highestY = r[1]
                screen = pygame.display.set_mode((int(highestX - lowestX), int(highestY - lowestY)))
                screen.fill("white")
                        
                for r in dots:
                    pygame.draw.circle(screen, "black", [r[0] - lowestX, r[1] - lowestY], 2)
                    
                pygame.display.update()
                
                pygame.image.save(screen, ("liveTest/move.jpeg"))
                lastMoves = getLabel(lastMoves)
        
def showData():
    
    fig, axLoss = plt.subplots()
    
    #df = p.chartDF(loss_list)
    #df = df[['close']]
    #df.reset_index(level = 0, inplace = True)
    #df.columns = ['ds']
    #m, b = np.polyfit(loss_list)
    
    color = 'tab:red'
    axLoss.set_xlabel('Epoch')
    axLoss.set_ylabel('Loss', color=color)
    axLoss.plot(loss_list, color=color)
    axLoss.tick_params(axis='y', labelcolor=color)
    
    #rolling_mean_axLoss = df.y.rolling(window = 20).mean()
    
    axAcc = axLoss.twinx()
    
    color = 'tab:blue'
    axAcc.set_ylabel('Test Accuracy', color=color)  
    axAcc.plot(test_accuracy, color=color)
    axAcc.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout() 
    #plt.plot(loss_list, m * loss_list + b)
    plt.show()
        
epochs = 100
for epoch in range(1, epochs):
    train(epoch)
    if(epoch % 10 == 0):
        test()
        print("Calculating...", str(int(epoch / epochs * 100)) + "% done!")
        
torch.save(model, "test.pt")

showData()
liveTest()