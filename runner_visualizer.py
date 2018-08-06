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
        ax = plt.gca()
        ylim_max = np.max(loss_data)
        ax.set_ylim(0, ylim_max+0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(axis='both', labelsize=14)
        plt.title('Training Loss Cost', fontsize=20)
        plt.ylabel('Cost', fontsize=20)
        plt.xlabel('Number of iterations (x100)', fontsize=20)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-8, y+0.5, f'{y:.2f}', fontsize=15)
        plt.show()

    @staticmethod
    def display_accuracy(training_accuracy, step=10):
        x = []
        y = []
        for i in range(0, len(training_accuracy), step):
            epoch = training_accuracy[i:i+step]
            x.append(np.argmax(epoch)+i)
            y.append(np.max(epoch))

        sns.set_style("white")
        plt.figure(figsize=(14,5))
        plt.plot(training_accuracy)
        plt.plot(x, y, 'o')
        ylim_min = np.min(training_accuracy)
        ylim_max = np.max(training_accuracy)
        plt.ylim(ylim_min-5, ylim_max+5)
        plt.tick_params(axis='both', labelsize=14)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title('Training Accuracy using 500 validations per evaluation', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Number of evaluations (1 every 1000 iterations)', fontsize=20)
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
        plt.title('One-shot (20-way) on 100 predictions using 50 validations per prediction', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Data Set', fontsize=20)
        ylim_min = np.min(predictions)
        ylim_max = np.max(predictions)
        plt.ylim(ylim_min-5, ylim_max+5)
        plt.show()