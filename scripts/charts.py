import matplotlib.pyplot as plt
from typing import List

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def plot_waterfall(expected_value_model: float, observation_values: List[float], 
                   observation_labels: List[str], contributions: List[float], 
                   name: str, index: int):
    """
    Creates a waterfall chart plot that visualizes the expected model value, 
    observation values, their contributions, and labels.

    Parameters:
        - expected_value_model (float): The expected model value.
        - observation_values (List[float]): A list of observation values.
        - observation_labels (List[str]): A list of labels associated with 
                                          the observation values.
        - contributions (List[float]): A list of contributions representing 
                                       changes from observation values to the 
                                       model value.
        - name (str): The name of the plot and the file for saving.
        - index (int): The index for the observation.
    """

    labels = [f'{value} = {label}' for (value, label) 
              in zip(observation_values, observation_labels)]
    waterfall = [expected_value_model]
    for contribution in contributions:
        waterfall.append(waterfall[-1] + contribution)

    _, ax = plt.subplots(figsize=(10, 3))

    for idx, (contribution, start) in enumerate(
        zip(contributions, waterfall[:-1])
    ):
        ax.barh(idx, contribution, left=start, color='blue' 
                if contribution >= 0 else 'red', height=0.5, alpha=0.8)
        text_x = start + contribution / 2
        ax.text(text_x, idx, contribution, va='center', ha='center', fontsize=12, color='white')

    l1 = r'$\mathbb{E}(f(X)) =$' +f' {expected_value_model} Euro'
    l2 = r'$f(x^{{({index})}})$ = {waterfall_value} Euro'.format(index=index, waterfall_value=waterfall[-1])
    ax.text(expected_value_model, -1.2, l1 , va='center', ha='center')
    ax.text(waterfall[-1], 2.8, l2, va='center', ha='center')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Euro')

    ax.axvline(x=expected_value_model, color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=waterfall[-1], color='grey', linestyle='--', linewidth=1)

    ax.set_xlim(min(waterfall)-5, max(waterfall)+5)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.tick_params(left = False)
    plt.savefig(f'images/{name}', dpi=300)

"""
Create charts for obersavtion x^{(1)} and x^{(2)}
"""
# General Model Values
observation_labels = ['Groesse', 'Anzahl Zimmer', 'Entfernung zum Zentrum']
expected_value_model = 680

# Observation values and contributions for x^{(1)}
index = 1
observation_values_x1 = [100, 3, 5]
contributions_x1 = [-125, -10, 5]

plot_waterfall(expected_value_model=expected_value_model, 
               observation_values=observation_values_x1, 
               observation_labels=observation_labels, 
               contributions=contributions_x1, 
               name="model-output-x1.png", index=index)

# Observation values and contributions for x^{(2)}
index = 2
observation_values_x2 = [150, 4, 10]
contributions_x2 = [125, 10, -5] 

plot_waterfall(expected_value_model=expected_value_model, 
               observation_values=observation_values_x2, 
               observation_labels=observation_labels, 
               contributions=contributions_x2, 
               name="model-output-x2.png", index=index)