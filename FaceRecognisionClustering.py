
from heapq import merge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.decomposition import KernelPCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import  classification_report 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from createDataset import *
warnings.filterwarnings('ignore')


#2 Importing the dataset

cluster_number = 10

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) , contingency_matrix

def pca_100():

    dataset = pd.read_csv("train_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #3 Applying Kernel PCA M=100
    kpca = KernelPCA(n_components = 100)
    X = kpca.fit_transform(X)
    df_pca_100 = pd.DataFrame(X)
    return df_pca_100

def pca_50():

    dataset = pd.read_csv("train_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #3 Applying Kernel PCA M=50
    kpca = KernelPCA(n_components = 50)
    X = kpca.fit_transform(X)
    df_pca_50 = pd.DataFrame(X)
    return df_pca_50

def pca_25():

    dataset = pd.read_csv("train_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #3 Applying Kernel PCA M=100
    kpca = KernelPCA(n_components = 25)
    X = kpca.fit_transform(X)
    df_pca_25 = pd.DataFrame(X)
    return df_pca_25


def K_Means(df_pca):
    dataset = pd.read_csv("train_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    kmeans = KMeans(n_clusters=cluster_number, init='k-means++')
    y_kmeans = kmeans.fit_predict(df_pca)
    df_pca["Class"] = y
    df_pca['y_kmeans'] = y_kmeans
    return df_pca

def FMeasure(df_pca ,contingency_matrix ):
   
    max_index = np.argmax(contingency_matrix,axis=1)
    list = []
    for x in df_pca["Class"]:
            if(x == 1):
                list.append(max_index[0])
            elif(x == 2):
                list.append(max_index[1])
            elif(x == 3):
                list.append(max_index[2])
            elif(x == 4):
                list.append(max_index[3])
            elif(x == 5):
                list.append(max_index[4])
            elif(x == 6):
                list.append(max_index[5])
            elif(x == 7):
                list.append(max_index[6])
            elif(x == 8):
                list.append(max_index[7])
            elif(x == 9):
                list.append(max_index[8])
            elif(x == 10):
                list.append(max_index[9])
    
    df_pca["Class"] = list


    report = classification_report(df_pca["Class"],df_pca['y_kmeans'] , output_dict=True)  
    total_F_Measure = 0

    for i in range(cluster_number):
        total_F_Measure += report[str(i)]['f1-score']

    return total_F_Measure

def HierarchicalClustering(df_pca):
    dataset = pd.read_csv("train_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    hc = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(df_pca)
    df_pca["Class"] = y
    df_pca['y_hc'] = y_hc

    return df_pca


from sklearn.utils import shuffle
def main():
    createDataset()

    df_pca_100 = pca_100()
    df_pca_50 = pca_50()
    df_pca_25 = pca_25()
#-------------------------------------------PCA 100--------------------------------------------------------------------------------------------

    pca_100_KMeans = K_Means(df_pca_100)
    df_pca_100_KMeans_purity_score , contingency_matrix = purity_score(pca_100_KMeans["Class"],pca_100_KMeans['y_kmeans'])
    print("\nPurity Score(KMeans , PCA_100):", df_pca_100_KMeans_purity_score)
    total_F_Measure_KMeans_pca_100 = FMeasure(pca_100_KMeans , contingency_matrix)
    print("Total_F_Measure(KMeans , PCA_100):", total_F_Measure_KMeans_pca_100)

    pca_100_HC = HierarchicalClustering(df_pca_100)
    df_pca_100_HC_purity_score , contingency_matrix = purity_score(pca_100_HC["Class"],pca_100_HC['y_kmeans'])
    print("\nPurity Score(HC , PCA_100):", df_pca_100_HC_purity_score)
    total_F_Measure_HC_pca_100 = FMeasure(pca_100_HC , contingency_matrix)
    print("Total_F_Measure(HC , PCA_100):", total_F_Measure_HC_pca_100)
    pca_100_HC.to_csv("Hierachical_PCA_100.csv")
#-------------------------------------------PCA 50--------------------------------------------------------------------------------------------

    pca_50_KMeans = K_Means(df_pca_50)
    df_pca_50_KMeans_purity_score , contingency_matrix = purity_score(pca_50_KMeans["Class"],pca_50_KMeans['y_kmeans'])
    print("\nPurity Score(KMeans , PCA_50):", df_pca_50_KMeans_purity_score)
    total_F_Measure_KMeans_pca_50 = FMeasure(pca_50_KMeans , contingency_matrix)
    print("Total_F_Measure(KMeans , PCA_50):", total_F_Measure_KMeans_pca_50)

    pca_50_HC = HierarchicalClustering(df_pca_50)
    df_pca_50_HC_purity_score , contingency_matrix = purity_score(pca_50_HC["Class"],pca_50_HC['y_kmeans'])
    print("\nPurity Score(HC , PCA_50):", df_pca_50_HC_purity_score)
    total_F_Measure_HC_pca_50 = FMeasure(pca_50_HC , contingency_matrix)
    print("Total_F_Measure(HC , PCA_50):", total_F_Measure_HC_pca_50)


#-------------------------------------------PCA 25--------------------------------------------------------------------------------------------

    pca_25_KMeans = K_Means(df_pca_25)
    df_pca_25_KMeans_purity_score , contingency_matrix = purity_score(pca_25_KMeans["Class"],pca_25_KMeans['y_kmeans'])
    print("\nPurity Score(KMeans , PCA_25):", df_pca_25_KMeans_purity_score)
    total_F_Measure_KMeans_pca_25 = FMeasure(pca_25_KMeans , contingency_matrix)
    print("Total_F_Measure(KMeans , PCA_25):", total_F_Measure_KMeans_pca_25)

    pca_25_HC = HierarchicalClustering(df_pca_25)
    df_pca_25_HC_purity_score , contingency_matrix = purity_score(pca_25_HC["Class"],pca_25_HC['y_kmeans'])
    print("\nPurity Score(HC , PCA_25):", df_pca_25_HC_purity_score)
    total_F_Measure_HC_pca_25 = FMeasure(pca_25_HC , contingency_matrix)
    print("Total_F_Measure(HC , PCA_25):", total_F_Measure_HC_pca_25)

if __name__ == '__main__':
   
    main()