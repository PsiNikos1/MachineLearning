import csv
import numpy as np
import os 
import sys

trainData=[]
testData=[]
submission_file=[]
k=1

def changeColorData():#white =0.0 , black=0.2 , clear=0.4 , blue=0.6 ,green =0.8 , blood=1
    rows = []
    fields=[]

    with open("test.csv", 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        fields = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    for i in range(len(rows)):
        if(rows[i][5] == "black"):
            rows[i][5] = 0.2
        elif(rows[i][5] == "white"):
            rows[i][5] = 0.0
        elif(rows[i][5] == "green"):
            rows[i][5] = 0.8
        elif(rows[i][5] == "blue"):
            rows[i][5] = 0.6
        elif(rows[i][5] == "blood"):
            rows[i][5] = 1.0
        elif(rows[i][5] == "clear"):
            rows[i][5] = 0.4    


    with open("test.csv",'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def createSubmissionFile():
    fields=["id" , "type"]

    with open("submission_file.csv",'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for header in fields:
            if header == fields[-1]:
                csvfile.write(str(header))
            else:
                csvfile.write(str(header)+',')
        csvfile.write('\n')
        for row in submission_file:
            for cell in row:
                if cell == row[-1]:
                    csvfile.write(str(cell) )
                else:
                    csvfile.write(str(cell)+',')
            csvfile.write('\n')

def readTrainData():
    fields=[]
    with open("train.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            trainData.append(row)

def readTestData():
    fields=[]
    with open("test.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            testData.append(row)

def findMonsterType(indexes , k):
    isGhoul=0
    isGhost=0
    isGoblin=0
    neighbors=0
    for index in indexes:
        neighbors+=1
        for rowNumber in range(len(trainData)):
            if(index == rowNumber ):
                #print(trainData[rowNumber]) # row are indexes + 2
                if(trainData[index][6] == "Ghoul"):
                    isGhoul+=1
                elif(trainData[index][6] == "Ghost"):
                    isGhost+=1
                elif(trainData[index][6] == "Goblin"):
                    isGoblin+=1
        if(neighbors>k-1):
            break

    #print(isGhoul , isGhost , isGoblin)                
    if(isGhoul > isGhost and isGhoul > isGoblin):
        return "Ghoul"
    elif(isGhoul < isGhost and isGhost > isGoblin):    
        return "Ghost"
    elif(isGoblin > isGhost and isGhost < isGoblin):    
        return "Goblin"
    elif(isGhost == isGoblin and isGoblin == isGhoul):
        return "Goblin" #TODO FIX THAT!
    else:
        return "Ghost"

def calcDistances(k):

    for i in range(len(testData)):#For every monster in test.
        id = testData[i][0]
        monsterType=""
        distances=[]
        for j in range(len(trainData)):#Calculate distance with every monster in train.
            test_point = np.array( (float(testData[i][1]) , float(testData[i][2]) , float(testData[i][3]) , float(testData[i][4]) , float(testData[i][5]) ) )
            train_point = np.array( (float(trainData[j][1]) , float(trainData[j][2]) , float(trainData[j][3]) , float(trainData[j][4]) , float(trainData[j][5]) )  )
            distances.append( np.linalg.norm(test_point - train_point) )
            #print(test_point)
            #print(train_point)
            
        indexes=findKNearestNeighbors(np.asarray(distances) , k)
        monsterType=findMonsterType(indexes , k)
        #print("ID:",id,"   Type:",monsterType)
        #print("--------------------------")
        submission_file.append( [id,monsterType] )

def findKNearestNeighbors( distances , k ):
    indexes = np.argpartition(distances, k)
    return indexes

def printResults():
    for monsters in submission_file:
        print(monsters)

def main():
    readTrainData()
    readTestData()
    calcDistances(k)
    createSubmissionFile()
    printResults()
main()
     