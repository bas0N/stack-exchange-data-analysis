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
    views_title_length_corr = posts_no_outliers['ViewCount'].corr(posts_no_outliers['TitleLength'])
    post_length_interactions_corr = posts_no_outliers['PostLength'].corr(posts_no_outliers['Interactions'])

    # Display individual correlations for the new DataFrame
    print(f"Correlation between views and interactions (outliers removed): {views_interactions_corr:.3f}")
    print(f"Correlation between views and post length (outliers removed): {views_post_length_corr:.3f}")
    print(f"Correlation between title length and interactions (outliers removed): {title_length_interactions_corr:.3f}")
    print(f"Correlation between views and title length (outliers removed): {views_title_length_corr:.3f}")
    print(f"Correlation between post length and interactions (outliers removed): {post_length_interactions_corr:.3f}")

    plt.figure(figsize=(25, 5))

    # Views to Interactions
    plt.subplot(1, 5, 1)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['Interactions'])
    plt.title('Views vs Interactions (Outliers Removed)')
    plt.text(0.05, 0.95, f'Corr: {views_interactions_corr:.3f}', transform=plt.gca().transAxes)

    # Views to Post Length
    plt.subplot(1, 5, 2)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['PostLength'])
    plt.title('Views vs Post Length (Outliers Removed)')
    plt.text(0.05, 0.95, f'Corr: {views_post_length_corr:.3f}', transform=plt.gca().transAxes)

    # Title Length vs Interactions
    plt.subplot(1, 5, 3)
    sns.scatterplot(x=posts_no_outliers['TitleLength'], y=posts_no_outliers['Interactions'])
    plt.title('Title Length vs Interactions (Outliers Removed)')
    plt.text(0.05, 0.95, f'Corr: {title_length_interactions_corr:.3f}', transform=plt.gca().transAxes)

    # Views to Title Length
    plt.subplot(1, 5, 4)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['TitleLength'])
    plt.title('Views vs Title Length (Outliers Removed)')
    plt.text(0.05, 0.95, f'Corr: {views_title_length_corr:.3f}', transform=plt.gca().transAxes)

    # Post Length vs Interactions
    plt.subplot(1, 5, 5)
    sns.scatterplot(x=posts_no_outliers['PostLength'], y=posts_no_outliers['Interactions'])
    plt.title('Post Length vs Interactions (Outliers Removed)')
    plt.text(0.05, 0.95, f'Corr: {post_length_interactions_corr:.3f}', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    # Return correlations
    return {
        'views_interactions_corr': views_interactions_corr,
        'views_post_length_corr': views_post_length_corr,
        'title_length_interactions_corr': title_length_interactions_corr,
        'views_title_length_corr': views_title_length_corr,
        'post_length_interactions_corr': post_length_interactions_corr
    }
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
    views_title_length_corr = posts_no_outliers['ViewCount'].corr(posts_no_outliers['TitleLength'])
    post_length_interactions_corr = posts_no_outliers['PostLength'].corr(posts_no_outliers['Interactions'])

    # Display individual correlations for the new DataFrame
    print(f"Correlation between views and interactions (outliers removed): {views_interactions_corr:.3f}")
    print(f"Correlation between views and post length (outliers removed): {views_post_length_corr:.3f}")
    print(f"Correlation between title length and interactions (outliers removed): {title_length_interactions_corr:.3f}")
    print(f"Correlation between views and title length (outliers removed): {views_title_length_corr:.3f}")
    print(f"Correlation between post length and interactions (outliers removed): {post_length_interactions_corr:.3f}")

    # Create scatter plots for visualizing individual correlations with outliers removed
    plt.figure(figsize=(25, 5))

    # Views to Interactions
    plt.subplot(1, 5, 1)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['Interactions'])
    plt.title('Views vs Interactions (Outliers Removed)')

    # Views to Post Length
    plt.subplot(1, 5, 2)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['PostLength'])
    plt.title('Views vs Post Length (Outliers Removed)')

    # Title Length vs Interactions
    plt.subplot(1, 5, 3)
    sns.scatterplot(x=posts_no_outliers['TitleLength'], y=posts_no_outliers['Interactions'])
    plt.title('Title Length vs Interactions (Outliers Removed)')

    # Views to Title Length
    plt.subplot(1, 5, 4)
    sns.scatterplot(x=posts_no_outliers['ViewCount'], y=posts_no_outliers['TitleLength'])
    plt.title('Views vs Title Length (Outliers Removed)')

    # Post Length vs Interactions
    plt.subplot(1, 5, 5)
    sns.scatterplot(x=posts_no_outliers['PostLength'], y=posts_no_outliers['Interactions'])
    plt.title('Post Length vs Interactions (Outliers Removed)')

    plt.tight_layout()
    plt.show()

    # Return correlations
    return {
        'views_interactions_corr': views_interactions_corr,
        'views_post_length_corr': views_post_length_corr,
        'title_length_interactions_corr': title_length_interactions_corr,
        'views_title_length_corr': views_title_length_corr,
        'post_length_interactions_corr': post_length_interactions_corr
    }



def plot_posts_over_time(posts_df):
    # Convert 'CreationDate' to datetime format
    posts_df['CreationDate'] = pd.to_datetime(posts_df['CreationDate'])

    # Extract year and month from 'CreationDate'
    posts_df['YearMonth'] = posts_df['CreationDate'].dt.to_period('M')

    # Group by YearMonth and count the number of posts
    posts_over_time = posts_df.groupby('YearMonth').size()

    # Plot the data
    posts_over_time.plot(kind='line', title='Number of Posts Over Time')
    plt.show()



def plot_top_tags(tags_df):
    # Convert 'Count' column to numeric, coerce errors to NaN
    tags_df['Count'] = pd.to_numeric(tags_df['Count'], errors='coerce')

    # Drop rows with NaN values in 'Count' column
    tags_df.dropna(subset=['Count'], inplace=True)

    # Get the top tags and their frequencies
    top_tags = tags_df.nlargest(10, 'Count')[['TagName', 'Count']]

    # Plot bar chart for top tags
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='TagName', data=top_tags, palette='viridis')
    plt.title('Top Tags and Their Frequencies')
    plt.xlabel('Tag Count')
    plt.ylabel('Tag Name')
    plt.show()


def plot_post_types_distribution(posts_df, posts_history_df):
    # Merge Posts and PostHistory DataFrames
    merged_posts_history = pd.merge(posts_df, posts_history_df, left_on='Id', right_on='PostId', how='inner')

    # Plot a countplot for PostTypeId
    plt.figure(figsize=(8, 6))
    sns.countplot(x='PostTypeId', data=merged_posts_history, palette='Set2')
    plt.title('Distribution of Post Types')
    plt.xlabel('Post Type ID')
    plt.ylabel('Count')
    plt.show()


def plot_reputation_comment_correlation(users_df, comments_ds):
    # Count comments per user
    comments_count = comments_ds['UserId'].value_counts().reset_index()
    comments_count.columns = ['UserId', 'CommentCount']
    comments_count['UserId'] = comments_count['UserId'].apply(pd.to_numeric, errors='coerce')

    # Convert relevant columns to numeric
    users_df['Id'] = users_df['Id'].apply(pd.to_numeric, errors='coerce')
    users_df['Reputation'] = users_df['Reputation'].apply(pd.to_numeric, errors='coerce')

    # Merge DataFrames
    merged_df = pd.merge(users_df, comments_count, left_on='Id', right_on='UserId', how='inner')

    # Calculate correlation
    correlation = merged_df['Reputation'].corr(merged_df['CommentCount'])

    # Handle outliers using IQR
    Q1 = merged_df['Reputation'].quantile(0.25)
    Q3 = merged_df['Reputation'].quantile(0.75)
    IQR = Q3 - Q1

    filtered_df = merged_df[(merged_df['Reputation'] >= Q1 - 1.5 * IQR) & (merged_df['Reputation'] <= Q3 + 1.5 * IQR)]

    # Calculate correlation for filtered data
    correlation_filtered = filtered_df['Reputation'].corr(filtered_df['CommentCount'])

    # Scatter plot
    plt.scatter(merged_df['Reputation'], merged_df['CommentCount'])
    plt.title('Correlation between User Reputation and Number of Comments')
    plt.xlabel('User Reputation')
    plt.ylabel('Number of Comments')
    plt.grid(True)

    # Display fewer points on the x-axis
    plt.xticks(ticks=range(0, int(max(merged_df['Reputation'])) + 1, 500), rotation=45)

    plt.show()

    print(f'Original Correlation: {correlation}')
    print(f'Filtered Correlation: {correlation_filtered}')