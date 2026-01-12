"""
Visualization & Report Generation Module
Untuk membuat charts dan laporan analisis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")


class VisualizationHelper:
    """
    Kelas untuk membuat visualisasi data
    """
    
    def __init__(self, style='seaborn', figsize=(12, 6)):
        """
        Inisialisasi visualization helper
        
        Args:
            style: Plot style
            figsize: Figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style)
            sns.set_palette("husl")
    
    def plot_sentiment_distribution(self, sentiments, title="Sentiment Distribution", output_path=None):
        """
        Plot sentiment distribution
        
        Args:
            sentiments: List or Series of sentiment labels
            title: Plot title
            output_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib is required for visualization")
            return None
        
        # Count sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot bar chart
        sentiment_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db'])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_word_frequency(self, word_freq_df, top_n=20, title="Top Words", output_path=None):
        """
        Plot word frequency
        
        Args:
            word_freq_df: DataFrame with 'word' and 'frequency' columns
            top_n: Number of top words to show
            title: Plot title
            output_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib is required for visualization")
            return None
        
        # Get top words
        top_words = word_freq_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        ax.barh(range(len(top_words)), top_words['frequency'].values, color='#3498db')
        
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words['word'].values)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(top_words['frequency'].values):
            ax.text(v + 0.5, i, str(v), va='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix", output_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
            output_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib is required for visualization")
            return None
        
        try:
            from sklearn.metrics import confusion_matrix
        except ImportError:
            print("⚠️  scikit-learn is required for confusion matrix")
            return None
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels if labels else range(cm.shape[0]),
                    yticklabels=labels if labels else range(cm.shape[0]))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_text_length_distribution(self, text_lengths, title="Text Length Distribution", output_path=None):
        """
        Plot text length distribution
        
        Args:
            text_lengths: List of text lengths
            title: Plot title
            output_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib is required for visualization")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram
        ax.hist(text_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Words', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add statistics
        mean_len = np.mean(text_lengths)
        median_len = np.median(text_lengths)
        ax.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.1f}')
        ax.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, metrics_df, title="Model Metrics Comparison", output_path=None):
        """
        Plot model metrics comparison
        
        Args:
            metrics_df: DataFrame with metrics
            title: Plot title
            output_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib is required for visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot grouped bar chart
        x = np.arange(len(metrics_df))
        width = 0.2
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(metrics):
            if metric in metrics_df.columns:
                ax.bar(x + (i * width), metrics_df[metric], width, label=metric.replace('_', ' ').title(), color=colors[i])
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig


class AnalysisReporter:
    """
    Kelas untuk generate laporan analisis
    """
    
    def __init__(self, output_dir='output'):
        """
        Inisialisasi report generator
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        
        # Create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_text_report(self, title, sections, output_filename='report.txt'):
        """
        Generate text report
        
        Args:
            title: Report title
            sections: Dict of section_name: content
            output_filename: Output filename
            
        Returns:
            str: Path to generated report
        """
        report_path = os.path.join(self.output_dir, output_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"{title.center(80)}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Sections
            for section_name, content in sections.items():
                f.write(f"\n{'─'*80}\n")
                f.write(f"[{section_name.upper()}]\n")
                f.write(f"{'─'*80}\n\n")
                
                if isinstance(content, pd.DataFrame):
                    f.write(content.to_string())
                elif isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(str(content))
                
                f.write("\n")
            
            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        return report_path
    
    def generate_excel_report(self, dataframes_dict, output_filename='report.xlsx'):
        """
        Generate Excel report dengan multiple sheets
        
        Args:
            dataframes_dict: Dict of sheet_name: DataFrame
            output_filename: Output filename
            
        Returns:
            str: Path to generated report
        """
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            print("⚠️  openpyxl is required for Excel export")
            return None
        
        report_path = os.path.join(self.output_dir, output_filename)
        
        # Write to Excel
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format sheet
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        return report_path
    
    def create_summary_stats(self, df, text_column, sentiment_column=None):
        """
        Create summary statistics
        
        Args:
            df: DataFrame
            text_column: Name of text column
            sentiment_column: Name of sentiment column (optional)
            
        Returns:
            dict: Summary statistics
        """
        text_lengths = [len(str(text).split()) for text in df[text_column].astype(str)]
        
        stats = {
            'Total Records': len(df),
            'Mean Text Length': round(np.mean(text_lengths), 2),
            'Median Text Length': int(np.median(text_lengths)),
            'Min Text Length': int(np.min(text_lengths)),
            'Max Text Length': int(np.max(text_lengths)),
            'Std Dev': round(np.std(text_lengths), 2)
        }
        
        if sentiment_column and sentiment_column in df.columns:
            stats['Sentiment Distribution'] = df[sentiment_column].value_counts().to_dict()
        
        return stats


# Testing
if __name__ == "__main__":
    print("="*80)
    print("VISUALIZATION TEST")
    print("="*80)
    
    if MATPLOTLIB_AVAILABLE:
        # Sample data
        sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative',
                      'positive', 'neutral', 'positive', 'negative', 'neutral']
        
        word_freq = pd.DataFrame({
            'word': ['tokopedia', 'bagus', 'tidak', 'bisa', 'cepat', 'lambat'],
            'frequency': [150, 120, 100, 95, 80, 70]
        })
        
        # Create visualizations
        viz = VisualizationHelper()
        
        print("\n[Creating visualizations...]")
        fig1 = viz.plot_sentiment_distribution(sentiments, title="Sentiment Distribution")
        print("✓ Sentiment distribution plot created")
        
        fig2 = viz.plot_word_frequency(word_freq, top_n=6, title="Top Words")
        print("✓ Word frequency plot created")
        
        # Generate reports
        print("\n[Creating reports...]")
        reporter = AnalysisReporter()
        
        sections = {
            'Summary': 'Sample analysis summary',
            'Statistics': pd.DataFrame({'Metric': ['Total', 'Unique'], 'Value': [10, 6]}),
            'Details': {'Accuracy': 0.95, 'F1-Score': 0.92}
        }
        
        report_path = reporter.generate_text_report(
            "Sample Analysis Report",
            sections,
            'sample_report.txt'
        )
        print(f"✓ Text report generated: {report_path}")
        
        dfs = {
            'Summary': word_freq,
            'Sentiments': pd.DataFrame({'Sentiment': sentiments})
        }
        excel_path = reporter.generate_excel_report(dfs, 'sample_report.xlsx')
        print(f"✓ Excel report generated: {excel_path}")
        
    else:
        print("⚠️  matplotlib/seaborn is required for visualization")
