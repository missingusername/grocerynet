import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_file = os.path.join('in', 'study', 'user_study.csv')

# Load the CSV file
df = pd.read_csv(data_file)

# Calculate the average values for each user
avg_df = df.groupby('User ID')[['Likert Model 1', 'Likert Model 2']].mean().reset_index()

# Rename user IDs to "Participant X"
avg_df['Participant'] = ['Participant {}'.format(i+1) for i in range(len(avg_df))]

# Melt the dataframe for seaborn
melted_df = avg_df.melt(id_vars='Participant', value_vars=['Likert Model 1', 'Likert Model 2'], 
                        var_name='Model', value_name='Average Likert Score')

# Rename the models in the melted dataframe
melted_df['Model'] = melted_df['Model'].replace({'Likert Model 1': 'Model 1', 'Likert Model 2': 'Model 2'})

# Create the grouped bar chart
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='Participant', y='Average Likert Score', hue='Model', data=melted_df)
plt.title('Average Likert Scores by Participant')
plt.xlabel('Participant')
plt.ylabel('Average Likert Score')
plt.xticks(rotation=45)
plt.legend(title='Model', loc='lower right')
plt.tight_layout()

# Add numbers on the bars
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 9),
                     textcoords = 'offset points')

plt.savefig(os.path.join('out', 'user_study.png'))

# Show the plot
plt.show()
