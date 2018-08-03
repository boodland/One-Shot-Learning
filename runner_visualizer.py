import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RunnerVisualizer:
    
    @staticmethod
    def display_loss(loss_data, step=100):
        sns.set_style("white")
        x = []
        y = []
        for i in range(0, len(loss_data), step):
            epoch = loss_data[i:i+step]
            x.append(np.argmin(epoch)+i)
            y.append(np.min(epoch))

        plt.figure(figsize=(14,6))
        plt.plot(loss_data)
        plt.plot(x, y, 'o')
        plt.yscale('log', basey=np.e)
        ax = plt.gca()
        y_axis_values = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 3, 4]
        ax.yaxis.set_ticks(y_axis_values)
        ax.set_ylim(0.05, 5)
        ax.set_yticklabels(y_axis_values)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(axis='both', labelsize=14)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-8, (np.log(y)+3)/10, f'{y:.2f}', fontsize=15)
        plt.show()

    @staticmethod
    def display_accuracy(training_accuracy):
        x = []
        y = []
        for i in range(0, len(training_accuracy), 10):
            epoch = training_accuracy[i:i+10]
            x.append(np.argmax(epoch)+i)
            y.append(np.max(epoch))

        sns.set_style("white")
        plt.figure(figsize=(14,5))
        plt.plot(training_accuracy)
        plt.plot(x, y, 'o')
        plt.ylim(60, 95)
        plt.tick_params(axis='both', labelsize=14)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-1.2, y+1.2, f'{y:.2f}', fontsize=15)
        plt.show()

    @staticmethod
    def display_predictions_accuracy(predictions):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10,5))
        ax = sns.boxplot(data= predictions, orient='v')
        ax.xaxis.set_ticklabels(['Train', 'Validation', 'Test'])
        ax.xaxis.set_ticks_position('none')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=15)
        plt.tick_params(axis='x', labelsize=20)
        plt.ylim(64, 101)
        plt.show()