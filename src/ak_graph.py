import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bar_graph():
    sns.set(style="whitegrid")

    hyperparameters = [3, 4, 5]

    tpr = [0, 0.91, 0.13]  
    fpr = [0, 0, 0]       

    df2 = pd.DataFrame({'Parameter': hyperparameters, 'Accuracy': tpr, 'Metric': 'TPR'})
    df3 = pd.DataFrame({'Parameter': hyperparameters, 'Accuracy': fpr, 'Metric': 'FPR'})

    df = pd.concat([df2, df3])

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Parameter', y='Accuracy', hue='Metric', data=df)
    bar_plot.set_title('Metric Comparison Over Hyperparameters')
    bar_plot.set_xlabel('Hyperparameter Levels')
    bar_plot.set_ylabel('Score')
    bar_plot.set_ylim(0, 1) 

    plt.legend(title='Metric')
    plt.show()

def plot_graph():
    sns.set(style="whitegrid")

    hyperparameters = [1, 2, 3, 4, 5]

    tpr = [0, 0, 0, 0.91, 0.13]  
    fpr = [0, 0, 0, 0, 0]        

    df2 = pd.DataFrame({'Parameter': hyperparameters, 'Accuracy': tpr, 'Metric': 'TPR'})
    df3 = pd.DataFrame({'Parameter': hyperparameters, 'Accuracy': fpr, 'Metric': 'FPR'})

    df = pd.concat([df2, df3])

    plt.figure(figsize=(10, 6))
    line_plot = sns.lineplot(x='Parameter', y='Accuracy', hue='Metric', data=df, marker='o')
    line_plot.set_title('Metric Comparison Over Hyperparameters')
    line_plot.set_xlabel('Hyperparameter Levels')
    line_plot.set_ylabel('Score')
    line_plot.set_ylim(0, 1)  

    plt.legend(title='Metric')
    plt.show()

if __name__ == "__main__":
    #plot_graph()
    plot_bar_graph()
