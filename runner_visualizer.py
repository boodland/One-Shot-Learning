import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RunnerVisualizer:
    
    @staticmethod
    def display_loss(loss_data, step=100):
        sns.set_style("white")
        x = []
        y = []
        total_data = len(loss_data)
        for i in range(0, total_data, step):
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
        plt.title('Training Loss Cost\nwith minimum cost per epoch (10000 iterations)\n', fontsize=20)
        plt.ylabel('Cost', fontsize=20)
        plt.xlabel('\nNumber of iterations (x100)', fontsize=20)
        x_offset = 2 * (total_data/step)
        y_offset = 0.25 * (total_data/step)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-x_offset, y+y_offset, f'{y:.2f}', fontsize=15)
        plt.show()

    @staticmethod
    def display_accuracy(training_accuracy, step=10):
        x = []
        y = []
        total_data = len(training_accuracy)
        for i in range(0, total_data, step):
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
        plt.title('Validation Accuracy using 50 validations per evaluation\nwith maximum accuracy per epoch (10000 iterations)\n', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('\nNumber of evaluations (1 every 1000 iterations)', fontsize=20)
        x_offset = 0.3 * (total_data/step)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-x_offset, y+1.5, f'{y:.2f}', fontsize=15)
        plt.show()

    @staticmethod
    def display_predictions(predictions):
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
        plt.title('One-shot 20-way task on 100 predictions\nusing 50 validations per prediction\n', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('\nData Set Types', fontsize=20)
        ylim_min = np.min(predictions)
        ylim_max = np.max(predictions)
        plt.ylim(ylim_min-5, ylim_max+5)
        plt.show()