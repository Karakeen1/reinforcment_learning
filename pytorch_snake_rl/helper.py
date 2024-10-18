import matplotlib.pyplot as plt
from IPython import display
import sys

plt.ion()
   
def plot(scores, mean_scores):
     # Redirect stdout to suppress figure output
    original_stdout = sys.stdout
    try:
        sys.stdout = open('nul', 'w')  # Use 'nul' for Windows
        display.clear_output(wait=True)
        plt.clf()  # Clear the current figure
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label='Scores')
        plt.plot(mean_scores, label='Mean Scores')
        plt.ylim(ymin=0)
        if scores:
            plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        if mean_scores:
            plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.legend()
        display.display(plt.gcf())  # Display the updated figure
        plt.pause(0.001)  # Pause to allow the figure to refresh
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout