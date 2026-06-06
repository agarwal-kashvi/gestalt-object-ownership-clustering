import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import argparse
import Levenshtein


class NewClusteringMetrics:
    def __init__(self, object_assignment_filename: str):
        """
        Initialize the clustering metrics calculator.
        
        Args:
            object_assignment_filename: Path to CSV file containing clustering results
        """
        self.metrics_df = pd.read_csv(object_assignment_filename)
        
        # Define true locations list
        self.true_locs_list = ['Ali\'s Vineyard', 'Little River Winery and Café', 
                              'Faber Vineyard', 'Ugly Duckling Wines', 
                              'Oakover Grounds', 'Lancaster Wines']
        
        # Clean the data
        self.metrics_df = self._clean_data()
        
        # Calculate TP and F_ based on Levenshtein ratio
        self.metrics_df.loc[:,'TP'] = self.metrics_df.apply(
            lambda row: Levenshtein.ratio(row['ground_truth_location'], row['predicted_location']) >= 0.7 
            if pd.notna(row['ground_truth_location']) and pd.notna(row['predicted_location']) 
            else False, 
            axis=1
        )
        self.metrics_df.loc[:,'F_'] = ~self.metrics_df['TP']
        
    def _clean_data(self) -> pd.DataFrame:
        """
        Clean the input data by handling missing values and removing invalid entries.
        """
        df = self.metrics_df.copy()
        
        # Remove rows where predicted_location is NaN or empty string
        df = df[df['predicted_location'].notna() & (df['predicted_location'] != '')]
        
        # Remove rows where cluster is -1 (unassigned)
        df = df[df['cluster'] != -1]
        
        return df
    
    def _calculate_cluster_sizes(self, location_type: str = 'predicted') -> Dict[str, int]:
        """
        Calculate the size of each cluster.
        
        Args:
            location_type: Either 'predicted' or 'ground_truth'
        """
        if location_type == 'predicted':
            return self.metrics_df.groupby('predicted_location').size().to_dict()
        else:
            return self.metrics_df.groupby('ground_truth_location').size().to_dict()
    
    def clustering_recall(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate recall for each true location and overall.
        
        Returns:
            Tuple containing (average recall, dictionary of recall per location)
        """
        TP_dict = {}
        FN_dict = {}
        recall_dict = {}
        
        for vineyard in self.true_locs_list:
            # Get all objects for this vineyard
            select_df = self.metrics_df[self.metrics_df['ground_truth_location'] == vineyard].drop_duplicates()
            select_df = select_df.sort_values(by='TP', ascending=False)
            select_df = select_df.drop_duplicates(subset=['longitude', 'latitude'], keep="first")
            
            TP_dict[vineyard] = len(select_df[select_df['TP'] == True])
            FN_dict[vineyard] = len(select_df[select_df['F_'] == True])
            
            # Calculate recall for this vineyard
            total = TP_dict[vineyard] + FN_dict[vineyard]
            if total > 0:
                recall_dict[vineyard] = TP_dict[vineyard] / total
            else:
                recall_dict[vineyard] = 0.0
        
        # Calculate average recall
        avg_recall = np.mean(list(recall_dict.values()))
        
        return avg_recall, recall_dict
    
    def clustering_weighted_recall(self) -> Tuple[float, Dict[str, float], Dict[str, int]]:
        """
        Calculate weighted recall for each true location and overall.
        
        Returns:
            Tuple containing (weighted average recall, dictionary of weighted recall per location, 
                            dictionary of class sizes)
        """
        TP_dict = {}
        FN_dict = {}
        weighted_recall_dict = {}
        total_dict = {}
        
        for vineyard in self.true_locs_list:
            # Get all objects for this vineyard
            select_df = self.metrics_df[self.metrics_df['ground_truth_location'] == vineyard].drop_duplicates()
            select_df = select_df.sort_values(by='TP', ascending=False)
            select_df = select_df.drop_duplicates(subset=['longitude', 'latitude'], keep="first")
            
            TP_dict[vineyard] = len(select_df[select_df['TP'] == True])
            FN_dict[vineyard] = len(select_df[select_df['F_'] == True])
            
            # Calculate recall and class size for this vineyard
            total = TP_dict[vineyard] + FN_dict[vineyard]
            if total > 0:
                recall = TP_dict[vineyard] / total
            else:
                recall = 0.0
                
            weighted_recall_dict[vineyard] = total * recall
            total_dict[vineyard] = total
        
        # Calculate weighted average recall
        total_objects = sum(total_dict.values())
        if total_objects > 0:
            weighted_avg_recall = sum(weighted_recall_dict.values()) / total_objects
        else:
            weighted_avg_recall = 0.0
        
        return weighted_avg_recall, weighted_recall_dict, total_dict
    
    def clustering_precision(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate precision for each predicted location and overall.
        
        Returns:
            Tuple containing (average precision, dictionary of precision per location)
        """
        TP_dict = {}
        FP_dict = {}
        precision_dict = {}
        
        for location in self.metrics_df['predicted_location'].unique():
            if pd.isna(location):
                continue
                
            # Get all objects for this location
            select_df = self.metrics_df[self.metrics_df['predicted_location'] == location].drop_duplicates()
            select_df = select_df.sort_values(by='TP', ascending=False)
            select_df = select_df.drop_duplicates(subset=['longitude', 'latitude'], keep="first")
            
            TP_dict[location] = len(select_df[select_df['TP'] == True])
            FP_dict[location] = len(select_df[select_df['F_'] == True])
            
            # Calculate precision for this location
            total = TP_dict[location] + FP_dict[location]
            if total > 0:
                precision_dict[location] = TP_dict[location] / total
            else:
                precision_dict[location] = 0.0
        
        # Calculate average precision
        avg_precision = np.mean(list(precision_dict.values()))
        
        return avg_precision, precision_dict
    
    def clustering_weighted_precision(self) -> Tuple[float, Dict[str, float], Dict[str, int]]:
        """
        Calculate weighted precision for each predicted location and overall.
        
        Returns:
            Tuple containing (weighted average precision, dictionary of weighted precision per location,
                            dictionary of class sizes)
        """
        TP_dict = {}
        FP_dict = {}
        weighted_precision_dict = {}
        total_dict = {}
        
        for location in self.metrics_df['predicted_location'].unique():
            if pd.isna(location):
                continue
                
            # Get all objects for this location
            select_df = self.metrics_df[self.metrics_df['predicted_location'] == location].drop_duplicates()
            select_df = select_df.sort_values(by='TP', ascending=False)
            select_df = select_df.drop_duplicates(subset=['longitude', 'latitude'], keep="first")
            
            TP_dict[location] = len(select_df[select_df['TP'] == True])
            FP_dict[location] = len(select_df[select_df['F_'] == True])
            
            # Calculate precision and class size for this location
            total = TP_dict[location] + FP_dict[location]
            if total > 0:
                precision = TP_dict[location] / total
            else:
                precision = 0.0
                
            weighted_precision_dict[location] = total * precision
            total_dict[location] = total
        
        # Calculate weighted average precision
        total_objects = sum(total_dict.values())
        if total_objects > 0:
            weighted_avg_precision = sum(weighted_precision_dict.values()) / total_objects
        else:
            weighted_avg_precision = 0.0
        
        return weighted_avg_precision, weighted_precision_dict, total_dict
    
    def print_metrics(self):
        """
        Print all clustering metrics in a formatted way.
        """
        print("\nClustering Metrics:")
        print("-" * 50)
        
        # Calculate and print recall metrics
        avg_recall, recall_dict = self.clustering_recall()
        print("\nRecall per location:")
        for location, recall in recall_dict.items():
            print(f"{location}: {recall:.4f}")
        print(f"\nAverage Recall: {avg_recall:.4f}")
        
        # Calculate and print weighted recall metrics
        weighted_recall, weighted_recall_dict, total_dict = self.clustering_weighted_recall()
        print("\nWeighted Recall per location:")
        for location, w_recall in weighted_recall_dict.items():
            print(f"{location}: {w_recall:.4f} (class size: {total_dict[location]})")
        print(f"\nWeighted Average Recall: {weighted_recall:.4f}")
        
        # Calculate and print precision metrics
        avg_precision, precision_dict = self.clustering_precision()
        print("\nPrecision per location:")
        for location, precision in precision_dict.items():
            print(f"{location}: {precision:.4f}")
        print(f"\nAverage Precision: {avg_precision:.4f}")
        
        # Calculate and print weighted precision metrics
        weighted_precision, weighted_precision_dict, total_dict = self.clustering_weighted_precision()
        print("\nWeighted Precision per location:")
        for location, w_precision in weighted_precision_dict.items():
            print(f"{location}: {w_precision:.4f} (class size: {total_dict[location]})")
        print(f"\nWeighted Average Precision: {weighted_precision:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate clustering metrics')
    parser.add_argument("-c", "--clusterFile",
                       help="CSV file containing clustering results",
                       type=str,
                       required=True)
    
    args = parser.parse_args()
    
    metrics = NewClusteringMetrics(args.clusterFile)
    metrics.print_metrics() 