import re

from collections import Counter 
import pandas as pd
import matplotlib.pyplot as plt



examples = [
    "They call it a Royale with cheese.",
    "A Royale with cheese. What do they call a Big Mac?",
    "Well, a Big Mac is a Big Mac, but they call it le Big-Mac.",
    "Le Big-Mac. Ha ha ha ha. What do they call a Whopper?"
]
language = set()
all_sentences= []
for sentence in examples:
    each_sentence = Counter()
    words= re.findall(r'\b\w+', sentence.lower())
    language.update(words)
    each_sentence.update(words)
    all_sentences.append(each_sentence)
    

kernel_matrices= {}
def calculate_score(all_sentences, d):
    score_table = [[0] * 4 for _ in range(4)]
    for i, mainsentence in enumerate(all_sentences, start=0):
        for j, comparedsentence in enumerate(all_sentences, start=0):
            for key, value in mainsentence.items():
                if key in comparedsentence:
                    score= value*comparedsentence[key]
                    score_table[i][j]=score_table[i][j]+score
            score_table[i][j]= score_table[i][j]**d
    kernel_matrices[d]= score_table
    

for d in [1,2,3,4]:
    calculate_score(all_sentences, d)

def pretty_print(kernel_matrices):
    for d, score_table in kernel_matrices.items():
        print(f"Kernel matrix for d = {d}:")
        score_df = pd.DataFrame(score_table, columns=["S1", "S2", "S3", "S4"], index=["S1", "S2", "S3", "S4"])
        print(score_df)
        print()
pretty_print(kernel_matrices)

def plot_kernel_matrices(kernel_matrices):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figure size if needed
    
    # Loop through each d value and corresponding score table
    for idx, (d, score_table) in enumerate(kernel_matrices.items()):
        ax = axes[idx]  # Select the current axis
        cax = ax.imshow(score_table, cmap='gray_r', interpolation='nearest')
        ax.set_title(f"d = {d}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["S1", "S2", "S3", "S4"])
        ax.set_yticks(range(4))
        ax.set_yticklabels(["S1", "S2", "S3", "S4"])
    
    # Add a color bar for the last plot, which will apply to all subplots
    fig.colorbar(cax, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, label='Score')
    plt.show()

# Call the plotting function
plot_kernel_matrices(kernel_matrices)












