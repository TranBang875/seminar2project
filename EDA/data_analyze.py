
"""Data Analyze.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install wordcloud
from wordcloud import WordCloud

df = pd.read_csv('https://raw.githubusercontent.com/TranBang875/seminar2project/main/dataset/train.csv')
display(df.head())

toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Melt the DataFrame to have one row per toxicity type per comment
melted_df = df.melt(id_vars=['id'],
                    value_vars=toxicity_types,
                    var_name='toxicity_type',
                    value_name='is_toxic')

# Filter for rows where 'is_toxic' is 1
toxic_distribution = melted_df[melted_df['is_toxic'] == 1]

display(toxic_distribution.head())

# Count the occurrences of each toxicity type
toxicity_counts = toxic_distribution['toxicity_type'].value_counts().reset_index()
toxicity_counts.columns = ['toxicity_type', 'count']

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='toxicity_type', y='count', data=toxicity_counts, palette='viridis')
plt.title('Distribution of Toxicity Types')
plt.xlabel('Toxicity Type')
plt.ylabel('Number of Comments')
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(toxicity_counts['count'], labels=toxicity_counts['toxicity_type'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(toxicity_counts)))
plt.title('Distribution of Toxicity Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"""Text-length"""

df['text_length'] = df['comment_text'].apply(len)
display(df.head())

melted_df_lengths = df.melt(id_vars=['id', 'text_length'],
                           value_vars=toxicity_types,
                           var_name='toxicity_type',
                           value_name='is_toxic')

display(melted_df_lengths.head())

toxic_melted_df_lengths = melted_df_lengths[melted_df_lengths['is_toxic'] == 1]
display(toxic_melted_df_lengths.head())

# Get the unique toxicity types
unique_toxicity_types = toxic_melted_df_lengths['toxicity_type'].unique()

# Create histograms for each toxicity type
for toxicity_type in unique_toxicity_types:
    plt.figure(figsize=(10, 6))
    subset_df = toxic_melted_df_lengths[toxic_melted_df_lengths['toxicity_type'] == toxicity_type]
    sns.histplot(data=subset_df, x='text_length', kde=True, color=sns.color_palette('viridis')[unique_toxicity_types.tolist().index(toxicity_type)])
    plt.title(f'Distribution of Text Lengths for {toxicity_type.capitalize()} Comments')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

plt.figure(figsize=(12, 7))
sns.histplot(data=toxic_melted_df_lengths, x='text_length', hue='toxicity_type', kde=True, palette='viridis', multiple='stack')
plt.title('Distribution of Text Lengths by Toxicity Type')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

"""Word_cloud"""

toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

from wordcloud import WordCloud

for toxicity_type in toxicity_types:
    # Filter the DataFrame for the current toxicity type
    toxic_comments = df[df[toxicity_type] == 1]['comment_text']

    # Concatenate all the comment text into a single string
    all_toxic_text = ' '.join(toxic_comments)

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_toxic_text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {toxicity_type.capitalize()} Comments')
    plt.show()
