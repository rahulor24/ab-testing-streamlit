import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Inspect the dataset for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.dropna()
    
    # Drop duplicate values
    df = df.drop_duplicates()
    
    # Outlier detection using IQR method for 'Total Ads' and 'Most Ads Hour' columns
    for column in ['total ads', 'most ads hour']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

class Visualization():

    def conversion_by_test_group(df):
        # Conversion by Test Group
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.countplot(data=df, x='test group', hue='converted',
                        palette=sns.color_palette("colorblind", 2))

        ax.set_xlabel("Test Group")
        ax.set_ylabel("Number of Users")
        ax.set_title("Converted vs Not Converted by Test Group")

        # Rename legend entries from True/False to readable labels
        handles, labels = ax.get_legend_handles_labels()
        label_map = {'True': 'Converted', 'False': 'Not Converted'}
        ax.legend(handles, [label_map.get(l, l) for l in labels], title='Conversion')

        # Annotate bars with counts
        for p in ax.patches:
            height = int(p.get_height())
            if height:
                ax.annotate(height,
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig


    # Visualize mean total ads by conversion status for each test group
    def mean_total_ads_by_conversion(df):
        fig, ax = plt.subplots(figsize=(8, 5))
        means_ad = df[df['test group']=='ad'].groupby('converted')['total ads'].mean()
        means_psa = df[df['test group']=='psa'].groupby('converted')['total ads'].mean()

        x = np.arange(2)
        width = 0.35

        ax.bar(x - width/2, means_ad.values, width, label='Ad Group')
        ax.bar(x + width/2, means_psa.values, width, label='PSA Group')

        ax.set_xticks(x)
        ax.set_xticklabels(['Not Converted', 'Converted'])
        ax.set_ylabel('Mean Total Ads')
        ax.set_title('Mean Total Ads by Conversion Status and Test Group')
        ax.legend()

        for i in range(2):
            ax.text(x[i] - width/2, means_ad.values[i], f"{means_ad.values[i]:.2f}", ha='center', va='bottom')
            ax.text(x[i] + width/2, means_psa.values[i], f"{means_psa.values[i]:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        return fig


    # Plot conversions by most ads day, bifurcated by test group
    def conversions_by_most_ads_day(df):
        conv = df[df['converted'] == True]
        counts = conv.groupby(['most ads day', 'test group']).size().unstack(fill_value=0)

        # enforce weekday order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        counts = counts.reindex(day_order).fillna(0)

        ax_conv = counts.plot(kind='bar', figsize=(10, 6), color=sns.color_palette("colorblind", 2))
        ax_conv.set_xlabel("Most Ads Day")
        ax_conv.set_ylabel("Number of Conversions")
        ax_conv.set_title("Conversions by Most Ads Day and Test Group")
        ax_conv.legend(title="Test Group")

        # annotate bars with counts
        for p in ax_conv.patches:
            h = p.get_height()
            if h:
                ax_conv.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h),
                                ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return ax_conv.get_figure()


    # Visualize mean most ads hour by conversion status for each test group
    def mean_most_ads_hour_by_conversion(df):
        fig, ax = plt.subplots(figsize=(8, 5))
        means_ad = df[df['test group']=='ad'].groupby('converted')['most ads hour'].mean()
        means_psa = df[df['test group']=='psa'].groupby('converted')['most ads hour'].mean()

        x = np.arange(2)
        width = 0.35

        ax.bar(x - width/2, means_ad.values, width, label='Ad Group')
        ax.bar(x + width/2, means_psa.values, width, label='PSA Group')

        ax.set_xticks(x)
        ax.set_xticklabels(['Not Converted', 'Converted'])
        ax.set_ylabel('Mean Most Ads Hours')
        ax.set_title('Mean Most Ads Hours by Conversion Status and Test Group')
        ax.legend()

        for i in range(2):
            ax.text(x[i] - width/2, means_ad.values[i], f"{means_ad.values[i]:.2f}", ha='center', va='bottom')
            ax.text(x[i] + width/2, means_psa.values[i], f"{means_psa.values[i]:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        return fig