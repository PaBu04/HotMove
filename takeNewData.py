'''
Created on 14.05.2021

@author: Paul Buda
'''

import pygame
import numpy as np
import win32api
import random

running = True

#Zaehlt alle je gespeicherte Pixel
number = 0
#Laenge der sichtbaren Pixelkette
length = 250

#Fenster initialisieren
screen = pygame.display.set_mode((100, 100))
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
            
            print("Save", number / 200)
            pygame.image.save(screen, ("new/random/random" + str(int(random.random() * 10 ** 15)) + ".jpeg"))