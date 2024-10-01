# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict

logger = logging.getLogger()


def generate_visualizations(attribute_weights: Dict[str, float]) -> None:
    """
    Generate and save a bar chart of attribute weights.
    """
    logger.info("Generating visualizations...")

    if not attribute_weights:
        logger.error("Attribute weights are empty, skipping visualization.")
        return

    primary_attrs = list(attribute_weights.keys())
    weights = list(attribute_weights.values())

    sns.set(style="whitegrid")
    plt.figure(figsize=(max(12, len(primary_attrs) * 1.5), 8))

    barplot = sns.barplot(x=primary_attrs, y=weights, palette="Blues_d")
    plt.xlabel("Primary Attributes", fontsize=12)
    plt.ylabel("Total Weight (%)", fontsize=12)
    plt.title("Relative Importance of Primary Customer Attributes", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Annotate bars with weights
    for index, bar in enumerate(barplot.patches):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{weights[index]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save chart
    chart_filename = "relative_importance_chart.png"
    plt.savefig(chart_filename, dpi=300)
    logger.info(f"Chart saved to {chart_filename}")
    # plt.show()  # Optional: Uncomment to display the plot
