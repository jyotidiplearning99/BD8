# save as metadata_analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_bd_s8_metadata():
    """Proper statistical analysis of your actual data"""
    
    # Load real data
    df = pd.read_excel('data/List of S8 data.xlsx')
    
    print("="*60)
    print("BD S8 Metadata Statistical Analysis")
    print("="*60)
    
    # 1. Descriptive Statistics
    print("\n📊 Sample Distribution:")
    print(df.groupby(['Extraction method', 'Fresh=1, frozen=0']).size())
    
    # 2. Chi-square test for associations
    print("\n📈 Statistical Tests:")
    
    # Test: Extraction method vs AB staining
    contingency = pd.crosstab(df['Extraction method'], df['AB stained=1'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(f"Extraction vs AB staining: χ²={chi2:.3f}, p={p_value:.3f}")
    
    # Test: Fresh/Frozen vs Viability
    if 'Viability dye=1' in df.columns:
        contingency2 = pd.crosstab(df['Fresh=1, frozen=0'], df['Viability dye=1'])
        chi2_2, p_value_2, dof_2, expected_2 = stats.chi2_contingency(contingency2)
        print(f"Fresh/Frozen vs Viability: χ²={chi2_2:.3f}, p={p_value_2:.3f}")
    
    # 3. Correlation matrix
    numeric_cols = ['Fresh=1, frozen=0', 'Fixed=1', 'Permeabilized=1', 
                    'Viability dye=1', 'AB stained=1']
    corr_matrix = df[numeric_cols].corr()
    
    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0,0], vmin=-1, vmax=1)
    axes[0,0].set_title('Feature Correlations')
    
    # Sample distribution
    df['Extraction method'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Extraction Method Distribution')
    axes[0,1].set_ylabel('Count')
    
    # Viability by preparation
    if 'Viability dye=1' in df.columns:
        viability_data = df.groupby('Fresh=1, frozen=0')['Viability dye=1'].mean()
        viability_data.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Viability by Fresh/Frozen Status')
        axes[1,0].set_ylabel('Proportion Viable')
        axes[1,0].set_xticklabels(['Fresh', 'Frozen'], rotation=0)
    
    # AB staining patterns
    ab_data = df.groupby(['Fixed=1', 'Permeabilized=1'])['AB stained=1'].mean()
    ab_data.unstack().plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('AB Staining by Fix/Perm Status')
    axes[1,1].set_ylabel('Proportion AB+')
    axes[1,1].legend(title='Permeabilized', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('outputs/metadata_analysis.png', dpi=150)
    plt.show()
    
    # 5. Generate report
    report = f"""
    BD S8 ANALYSIS REPORT
    ====================
    
    Dataset: n={len(df)} samples
    AB Stained: {df['AB stained=1'].sum()} samples
    
    Key Findings:
    -------------
    1. Extraction Method Effect:
       - Ficoll: {len(df[df['Extraction method']=='Ficoll'])} samples
       - Buffy coat: {len(df[df['Extraction method']=='Buffy coat'])} samples
       - Association with AB staining: p={p_value:.3f}
    
    2. Sample Quality:
       - Fresh samples: {len(df[df['Fresh=1, frozen=0']==1])}
       - Frozen samples: {len(df[df['Fresh=1, frozen=0']==0])}
       - Fixed samples: {df['Fixed=1'].sum()}
       - Permeabilized: {df['Permeabilized=1'].sum()}
    
    3. Correlations:
    {corr_matrix.to_string()}
    
    Recommendations:
    ---------------
    - Sample size too small for deep learning (n=3 with images)
    - Request TIFF conversion for image analysis
    - Focus on metadata patterns for preliminary insights
    """
    
    print(report)
    
    with open('outputs/analysis_report.txt', 'w') as f:
        f.write(report)
    
    return df, corr_matrix

# Run the analysis
if __name__ == "__main__":
    df, correlations = analyze_bd_s8_metadata()
