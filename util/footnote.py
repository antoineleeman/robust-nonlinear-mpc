from datetime import datetime


def add_footnote_time(plt):
    # Add the current date and time to the bottom-right of the plot
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated on: {current_time}", ha="right", fontsize=8)
