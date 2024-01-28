import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_comment_times(comments):
    # Convert 'CreationDate' to datetime
    comments['CreationDate'] = pd.to_datetime(comments['CreationDate'])

    # Extract the hour and date from 'CreationDate'
    comments['Hour'] = comments['CreationDate'].dt.hour
    comments['Date'] = comments['CreationDate'].dt.date

    # Group by date and hour, then count the number of comments
    grouped = comments.groupby(['Date', 'Hour']).size().reset_index(name='CommentCount')

    # Calculate average comment count per hour (normalizing the data)
    avg_comments_per_hour = grouped.groupby('Hour')['CommentCount'].mean().reset_index()

    # Plotting with Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Hour', y='CommentCount', data=avg_comments_per_hour, palette="coolwarm")
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Comments')
    plt.title('Average Number of Comments per Hour')
    plt.xticks(range(0, 24))
    plt.grid(axis='y')
    plt.show()


def analyze_votes_and_engagement(votes, posts):
    # Vote type mapping
    vote_type_mapping = {
        1: 'AcceptedByOriginator',
        2: 'UpMod',
        3: 'DownMod',
        4: 'Offensive',
        5: 'Favorite',
        6: 'Close',
        7: 'Reopen',
        8: 'BountyStart',
        9: 'BountyClose',
        10: 'Deletion',
        11: 'Undeletion',
        12: 'Spam',
        15: 'ModeratorReview',
        16: 'ApproveEditSuggestion'
    }

    # Replace VoteTypeId with names
    votes['VoteTypeName'] = votes['VoteTypeId'].map(vote_type_mapping)

    # Filter relevant columns
    vote_counts = votes[['PostId', 'VoteTypeName']]
    posts_interactions = posts[['Id', 'AnswerCount', 'CommentCount', 'FavoriteCount']]
    
    # Calculate total engagement for each post
    posts_interactions['Interactions'] = posts_interactions['AnswerCount'] + posts_interactions['CommentCount'] + posts_interactions['FavoriteCount']

    # Aggregate vote counts by PostId and VoteTypeName
    vote_counts = vote_counts.groupby(['PostId', 'VoteTypeName']).size().unstack(fill_value=0)

    # Merge vote counts with post interactions
    merged_data = pd.merge(vote_counts, posts_interactions, left_on='PostId', right_on='Id')

    # Correlation analysis
    correlation = merged_data.corr()

    # Visualization: Correlation Heatmap
    plt.figure(figsize=(16, 12)) # Increased figure size
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 8}) # Adjusted annotation format and size
    plt.title('Correlation between Vote Types and Post Interactions')
    plt.show()

    return correlation
def analyze_user_reputation_and_post_interactions(users, posts):
    # Assuming 'Id' and 'Reputation' are the correct column names in users
    user_reputation = users[['Id', 'Reputation']].rename(columns={'Id': 'UserId'})

    # Adjusting to the correct column names in posts
    # Replace 'PostId' with 'Id' and 'UserId' with 'OwnerUserId' or appropriate names
    post_interactions = posts[['Id', 'OwnerUserId', 'ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount']]
    post_interactions = post_interactions.rename(columns={'Id': 'PostId', 'OwnerUserId': 'UserId'})

    # Calculate total interactions for each post
    post_interactions['TotalInteractions'] = post_interactions[['ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount']].sum(axis=1)

    # Merge user reputation with post interactions
    merged_data = pd.merge(post_interactions, user_reputation, on='UserId')

    # Correlation analysis
    correlation = merged_data[['Reputation', 'TotalInteractions']].corr()

    # Visualization: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation between User Reputation and Post Interactions')
    plt.show()

    return correlation
def analyze_comments_by_time_of_year(comments):
    """
    Analyze the number of comments made in each part of the year.

    :param comments: DataFrame containing comments data with a column for creation dates.
    """
    # Convert 'CreationDate' to datetime
    comments['CreationDate'] = pd.to_datetime(comments['CreationDate'])

    # Extract month and year from 'CreationDate'
    comments['Year'] = comments['CreationDate'].dt.year
    comments['Month'] = comments['CreationDate'].dt.month

    # Group by year and month, then count the number of comments
    monthly_comments = comments.groupby(['Year', 'Month']).size().reset_index(name='CommentCount')

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Month', y='CommentCount', hue='Year', data=monthly_comments, marker='o')
    plt.xlabel('Month of Year')
    plt.ylabel('Number of Comments')
    plt.title('Monthly Comment Trends Over Years')
    plt.xticks(range(1, 13))
    plt.legend(title='Year')
    plt.show()

    return monthly_comments

def analyze_post_data(posts):
    # Convert numeric columns to appropriate data types for Posts
    numeric_columns = ['ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount']
    posts[numeric_columns] = posts[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Interactions will be the sum of AnswerCount, CommentCount, and FavoriteCount
    posts['Interactions'] = posts['AnswerCount'] + posts['CommentCount'] + posts['FavoriteCount']

    # Calculate post length as the length of the Body
    posts['PostLength'] = posts['Body'].str.len()

    # Calculate title length as the length of the Title
    posts['TitleLength'] = posts['Title'].str.len()

    # Drop rows with missing data that are essential for correlation analysis
    posts.dropna(subset=['ViewCount', 'Interactions', 'PostLength', 'TitleLength'], inplace=True)
    # Calculate z-scores for the relevant columns to identify outliers
    z_scores = stats.zscore(posts[['ViewCount', 'Interactions', 'PostLength', 'TitleLength']])

    # Define a threshold for identifying outliers (e.g., z-score greater than 3)
    threshold = 3

    # Create a mask to filter out outliers
    outlier_mask = (z_scores < threshold).all(axis=1)

    # Create a new DataFrame with outliers removed
    posts_no_outliers = posts[outlier_mask]

    # Calculate correlations for the new DataFrame
    views_interactions_corr = posts_no_outliers['ViewCount'].corr(posts_no_outliers['Interactions'])
    views_post_length_corr = posts_no_outliers['ViewCount'].corr(posts_no_outliers['PostLength'])
    title_length_interactions_corr = posts_no_outliers['TitleLength'].corr(posts_no_outliers['Interactions'])

    # Display individual correlations for the new DataFrame
    print(f"Correlation between views and interactions (outliers removed): {views_interactions_corr:.3f}")
    print(f"Correlation between views and post length (outliers removed): {views_post_length_corr:.3f}")
    print(f"Correlation between title length and interactions (outliers removed): {title_length_interactions_corr:.3f}")

    # Create scatter plots for visualizing individual correlations with outliers removed
    plt.figure(figsize=(15, 5))

    # Views to Interactions
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['Interactions'])
    plt.title('Views vs Interactions (Outliers Removed)')

    # Views to Post Length
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['PostLength'])
    plt.title('Views vs Post Length (Outliers Removed)')

    # Title Length vs Interactions
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=posts_no_outliers['TitleLength'], y=posts_no_outliers['Interactions'])
    plt.title('Title Length vs Interactions (Outliers Removed)')

    plt.tight_layout()
    plt.show()

    # Return correlations
    return {
        'views_interactions_corr': views_interactions_corr,
        'views_post_length_corr': views_post_length_corr,
        'title_length_interactions_corr': title_length_interactions_corr
    }
    