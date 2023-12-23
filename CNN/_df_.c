from matplotlib import pyplot as plt
import seaborn as sns
_df_7 = pd.DataFrame({'participant_id': [1, 2, 3, 4, 5], 'true_label': [0, 1, 0, 1, 0]})
figsize = (12, 1.2 * len(_df_7['participant_id'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(_df_7, x='true_label', y='participant_id', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)


from matplotlib import pyplot as plt
_df_6 = pd.DataFrame({'true_label': [1, 2, 3, 4, 5]})
_df_6['true_label'].plot(kind='line', figsize=(8, 4), title='true_label')
plt.gca().spines[['top', 'right']].set_visible(False)


from matplotlib import pyplot as plt
import seaborn as sns
_df_7.groupby('participant_id').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)


from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
_df_4 = pd.DataFrame({'true_label': [1, 2, 3, 4, 5]})
_df_4['true_label'].plot(kind='hist', bins=20, title='true_label')
plt.gca().spines[['top', 'right',]].set_visible(False)

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob

# Find all CSV files in the current working directory
csv_files = glob.glob('*.csv')

# Check if there are any CSV files
if len(csv_files) == 0:
  print('No CSV files found.')
else:
  # Read the first CSV file
  _df_5 = pd.read_csv(csv_files[0])
  _df_5.groupby('participant_id').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
  plt.gca().spines[['top', 'right',]].set_visible(False)
