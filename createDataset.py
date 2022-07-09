import numpy as np
import random
import os
from tokenize import Double
from PIL import Image
import pandas as pd

directory = 'C:\\Users\\nikos\\Desktop\\MachineLearning\\FaceClassification\\train_data'

def imageProcession(path): #Needs the path of a photo , returns r g b values for the photo and calculates the 1-color vector
    color_img = np.asarray(Image.open(path)) / 255    #  "C:\\Users\\nikos\\Desktop\\MachineLearning\\FaceClassification\\test.jpg"
    list = []



    for x in color_img:
        for y in x:
            val = colorTransformFormula(y[0], y[1], y[2])
            list.append(val)
    
    #print(list)
    return list   

def colorTransformFormula(r, g, b):
    val = 0.299 * r + 0.587 * g + 0.114 * b
    return val

def createIndividuals(directory): 
    random_dirs= random.sample(range(0, 3999), 10) #10 random people
  
    #directory = 'train_data'
    unique_people_photos = []
   
    for random_dir in random_dirs: #For each random people
        files = os.listdir(directory + "\\" + str(random_dir)) # file in directory
        person_path = directory + "\\" +str(random_dir) +"\\"

        if(len(files) > 50): #if the directory has more than 50 files.
            unique_photo_paths = []
           
            for i in range(0,50): #Choose 50 random photos
                photo_path =  person_path + random.choice(os.listdir(directory + "\\" + str(random_dir)) )
                #print(photo_path)

                while(photo_path  in unique_photo_paths):
                    #print(photo_path)

                    photo_path =  person_path + random.choice(os.listdir(directory + "\\" + str(random_dir)) )  
                unique_photo_paths.append( photo_path)
            unique_people_photos.append(unique_photo_paths)
       
        else: #if the directory has less than 50 files.
            unique_photo_paths = []
            for dir in files: #Choose  them all
                unique_photo_paths.append(person_path + "\\" +str(dir))
                #print(directory + "\\" +str(dir))
            unique_people_photos.append(unique_photo_paths)
    
    return unique_people_photos

def createDataset():
    print("Creating dataset....")
    unique_individuals =  createIndividuals(directory)
    processed_individuals = [] #This has 10 individuals,1 for each person, and each individual has 50 elements , one for each photo.  Every one element is a list with size=64x64 with all one_color coding.

    for person in unique_individuals:
        processed_photos=[]
        for photo in person:
            #print(photo)
            processed_photo = imageProcession(photo)
            processed_photos.append(processed_photo)
        processed_individuals.append(processed_photos)

    data =[]
    ind = 0
    for per in  processed_individuals:
        ind += 1
        for photo in per:#runs for  64x64 times
            list = []
            for vec in photo:
                list.append(vec)
            list.append(ind)
            data.append(list)
    df = pd.DataFrame(data)
    #print(df.head())
    #"C:\\Users\\nikos\Desktop\\MachineLearning\\FaceClassification\\train_data.csv"
    #df = df.sample(frac = 1)
    df.to_csv("train_data.csv")
    print("Dataset created")

    #print(len(processed_individuals)) # Number of people
    #print(len(processed_individuals[0])) # Number of photos of every person
    #print(processed_individuals[0][1]) # Number of vectors in each photo



if __name__ == '__main__':
    createDataset()