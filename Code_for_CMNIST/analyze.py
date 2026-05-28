import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

# sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()
    
class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str="", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, linewidth=2)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=20)
    plt.xlabel("Radius", fontsize=20)
    plt.ylabel("Certified Accuracy", fontsize=20)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16, bbox_to_anchor=(1, 1), borderaxespad=0.1)
    plt.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.3)
    current_yticks = plt.yticks()[0] 
    new_yticks = current_yticks[current_yticks != 0]
    plt.yticks(new_yticks)  
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    # plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()

def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()

def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.2f} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.3f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.3f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()

if __name__ == "__main__":

    # Example of how to use plot_certified_accuracy
    
    plot_certified_accuracy(
        "Figure_0.12", "Result of CMNIST", 0.5, [
            Line(ApproximateAccuracy("./output_file"), "Method1"),
            # Line(ApproximateAccuracy("./output_file2"), "Method2"),
        ])
    
    