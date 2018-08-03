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
        ax.set_ylim(0.07, 4)
        ax.set_yticklabels(y_axis_values)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(axis='both', labelsize=14)
        for xy in zip(x, y):
            x, y = xy
            ax.text(x-8, (np.log(y)+3)/10, f'{y:.2f}', fontsize=15)
        plt.show()