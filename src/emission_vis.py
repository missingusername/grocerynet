import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def read_csv_files_from_folder(folder_path, task_column, task_name, value_column):
    data = []
    filenames = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Sort files based on numerical prefix
    files.sort(key=lambda f: int(re.match(r'(\d+)', f).group()))
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        if task_column in df.columns and value_column in df.columns:
            filtered_row = df[df[task_column] == task_name]
            if not filtered_row.empty:
                data.append(filtered_row[value_column].values[0] * 1000)  # Convert to grams
                filenames.append(filename)
    return filenames, data

def generate_bar_chart(filenames, data, value_column):
    barplot = sns.barplot(x=filenames, y=data)
    plt.title(f'CO2-equivalent (grams) during training of each model')
    plt.xlabel('Files')
    plt.ylabel(f'Value of {value_column} (grams)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add numbers on the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center',
                         xytext = (0, 9),
                         textcoords = 'offset points')

    plt.savefig('out/epoch_emissions.png')
    plt.show()

def main(folder_path, task_column, task_name, value_column):
    filenames, data = read_csv_files_from_folder(folder_path, task_column, task_name, value_column)
    generate_bar_chart(filenames, data, value_column)

if __name__ == "__main__":
    folder_path = os.path.join('in', 'emissions')
    task_column = 'task_name'
    task_name = 'train model'
    value_column = 'emissions'
    main(folder_path, task_column, task_name, value_column)
