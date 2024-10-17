import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_exp_path = '../../dataset/df_exp_merged.csv'  # Replace with your actual file path
df_mp_path = '../../dataset/df_mp_merged.csv'    # Replace with your actual file path

df_exp = pd.read_csv(df_exp_path)
df_mp = pd.read_csv(df_mp_path)

# Function to find and plot top 10 correlated features
def find_top_correlated_features(df, dataset_name):
    # Drop the 'formula' column as it's not needed for correlation analysis
    df = df.drop(columns=['formula'])
    
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    
    # Get the correlation of all features with the target
    target_correlation = correlation_matrix['target']
    
    # Remove the target correlation with itself
    target_correlation = target_correlation.drop(labels=['target'])
    
    # Get the top 10 features with the highest correlation with the target
    top_10_features = target_correlation.abs().sort_values(ascending=False).head(10)
    
    # Get the top 10 feature names and their correlation values
    top_10_feature_names = top_10_features.index
    top_10_feature_values = top_10_features.values
    
    # Plot the top 10 features with the highest correlation with the target
    fig = plt.figure(figsize=(12, 4.8))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    left, bottom, width, height = 0.5, 0.15, 0.45, 0.75  # 位置とサイズをFigureの幅と高さに対する比率で指定
    ax = fig.add_axes([left, bottom, width, height])
    plt.barh(top_10_feature_names, top_10_feature_values)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Absolute correlation coefficient with band gap',fontsize=23)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.0)
    if dataset_name == "df_exp": 
        plt.title("Experimental dataset", fontsize=25)
    else: 
        plt.title("MP dataset", fontsize=25)
    
    plt.tight_layout()
    plt.savefig(f'img/{dataset_name}_top_10_features_correlated_with_target_updated.png')
    
    # Save the top 10 features to a CSV file
    output_df = pd.DataFrame({
        'Feature': top_10_feature_names,
        'Correlation': top_10_feature_values
    })
    
    output_csv_path = f'csv/{dataset_name}_top_10_features_correlated_with_target_updated.csv'
    output_df.to_csv(output_csv_path, index=False)
    
    return output_df, output_csv_path

# Find top correlated features for both datasets
exp_top_10_features, exp_csv_path = find_top_correlated_features(df_exp, 'df_exp')
mp_top_10_features, mp_csv_path = find_top_correlated_features(df_mp, 'df_mp')



print(f'Top 10 features CSV for df_exp saved to: {exp_csv_path}')
print(f'Top 10 features CSV for df_mp saved to: {mp_csv_path}')
