

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class FaceRecognitionAnalytics:
    def __init__(self, db_path="attendance.db"):
        self.db_path = db_path
        self.init_analytics_db()
        self.test_results = []
        
    def init_analytics_db(self):
        """Initialize analytics database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test results table for storing individual recognition attempts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                actual_student_id TEXT,
                predicted_student_id TEXT,
                confidence_score REAL,
                lighting_condition TEXT,
                distance_from_camera REAL,
                face_angle TEXT,
                is_correct_identification BOOLEAN,
                is_accepted BOOLEAN,
                threshold_used REAL
            )
        ''')
        
        # System performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_session_id TEXT,
                timestamp DATETIME,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                false_acceptance_rate REAL,
                false_rejection_rate REAL,
                equal_error_rate REAL,
                total_tests INTEGER,
                lighting_condition TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_recognition_attempt(self, actual_id, predicted_id, confidence, 
                              lighting_condition="normal", distance=2.0, 
                              face_angle="frontal", threshold=0.9):
        """Log a single recognition attempt for later analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        is_correct = (actual_id == predicted_id)
        # Probability-style acceptance: higher confidence is better (e.g., 0..1)
        is_accepted = (confidence >= threshold)
        
        cursor.execute('''
            INSERT INTO test_results 
            (timestamp, actual_student_id, predicted_student_id, confidence_score,
             lighting_condition, distance_from_camera, face_angle, 
             is_correct_identification, is_accepted, threshold_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), actual_id, predicted_id, confidence,
              lighting_condition, distance, face_angle, is_correct, int(is_accepted), threshold))
        
        conn.commit()
        conn.close()
    
    def calculate_metrics_from_db(self, lighting_condition=None, days_back=30):
        """Calculate all metrics from stored test results"""
        conn = sqlite3.connect(self.db_path)
        
        # Build query based on filters
        query = '''
            SELECT actual_student_id, predicted_student_id, confidence_score,
                   is_correct_identification, is_accepted, threshold_used
            FROM test_results 
            WHERE timestamp >= ?
        '''
        params = [datetime.now() - timedelta(days=days_back)]
        
        if lighting_condition:
            query += ' AND lighting_condition = ?'
            params.append(lighting_condition)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
        
        return self.calculate_metrics_from_dataframe(df)
    
    def calculate_metrics_from_dataframe(self, df):
        """Calculate all metrics from a pandas DataFrame"""
        total_tests = len(df)
        
        if total_tests == 0:
            return None
        
        # Convert boolean-like columns to real bools
        # df['is_correct_identification'] = df['is_correct_identification'].astype(str).str.strip().replace({'00': '0', '01': '1'})
        # df['is_accepted'] = df['is_accepted'].astype(str).str.strip().replace({'00': '0', '01': '1'})

        # df['is_correct_identification'] = df['is_correct_identification'].astype(int).astype(bool)
        # df['is_accepted'] = df['is_accepted'].astype(int).astype(bool)

        # Basic counts
        true_positives = len(df[(df['is_correct_identification'] == True) & 
                               (df['is_accepted'] == True)])
        false_positives = len(df[(df['is_correct_identification'] == False) & 
                                (df['is_accepted'] == True)])
        true_negatives = len(df[(df['is_correct_identification'] == False) & 
                               (df['is_accepted'] == False)])
        false_negatives = len(df[(df['is_correct_identification'] == True) & 
                                (df['is_accepted'] == False)])
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total_tests if total_tests > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # False Acceptance Rate (FAR) - accepting wrong person
        total_negatives = len(df[df['actual_student_id'] != df['predicted_student_id']])
        far = false_positives / total_negatives if total_negatives > 0 else 0
        
        # False Rejection Rate (FRR) - rejecting correct person
        total_positives = len(df[df['actual_student_id'] == df['predicted_student_id']])
        frr = false_negatives / total_positives if total_positives > 0 else 0
        
        # Equal Error Rate (EER) - calculate by varying threshold
        eer = self.calculate_eer(df)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'false_acceptance_rate': far,
            'false_rejection_rate': frr,
            'equal_error_rate': eer,
            'total_tests': total_tests,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
        
        return metrics
    
    def calculate_eer(self, df):
        """Calculate Equal Error Rate by varying threshold"""
        if df.empty:
            return 0
        
        # Get range of confidence scores
        min_conf = df['confidence_score'].min()
        max_conf = df['confidence_score'].max()
        
        thresholds = np.linspace(min_conf, max_conf, 100)
        far_rates = []
        frr_rates = []
        
        for threshold in thresholds:
            # Recalculate acceptance based on threshold
            df_thresh = df.copy()
            df_thresh['is_accepted_thresh'] = df_thresh['confidence_score'] >= threshold
            
            # Calculate FAR and FRR for this threshold
            false_accepts = len(df_thresh[(df_thresh['is_correct_identification'] == False) & 
                                        (df_thresh['is_accepted_thresh'] == True)])
            total_negatives = len(df_thresh[df_thresh['is_correct_identification'] == False])
            far = false_accepts / total_negatives if total_negatives > 0 else 0
            
            false_rejects = len(df_thresh[(df_thresh['is_correct_identification'] == True) & 
                                        (df_thresh['is_accepted_thresh'] == False)])
            total_positives = len(df_thresh[df_thresh['is_correct_identification'] == True])
            frr = false_rejects / total_positives if total_positives > 0 else 0
            
            far_rates.append(far)
            frr_rates.append(frr)
        
        # Find point where FAR and FRR are closest
        differences = [abs(far - frr) for far, frr in zip(far_rates, frr_rates)]
        min_diff_idx = differences.index(min(differences))
        
        eer = (far_rates[min_diff_idx] + frr_rates[min_diff_idx]) / 2
        return eer
    
    def generate_performance_report(self, lighting_condition=None, days_back=30):
        """Generate a comprehensive performance report"""
        metrics = self.calculate_metrics_from_db(lighting_condition, days_back)
        
        if not metrics:
            return "No test data available for the specified period."
        
        report = f"""
FACIAL RECOGNITION SYSTEM PERFORMANCE REPORT
============================================
Test Period: Last {days_back} days
Lighting Condition: {lighting_condition or 'All conditions'}
Total Tests: {metrics['total_tests']}

CORE METRICS:
- Accuracy: {metrics['accuracy']:.2%}
- Precision: {metrics['precision']:.2%}
- Recall (Sensitivity): {metrics['recall']:.2%}
- False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.2%}
- False Rejection Rate (FRR): {metrics['false_rejection_rate']:.2%}
- Equal Error Rate (EER): {metrics['equal_error_rate']:.2%}

CONFUSION MATRIX:
- True Positives: {metrics['true_positives']}
- False Positives: {metrics['false_positives']}
- True Negatives: {metrics['true_negatives']}
- False Negatives: {metrics['false_negatives']}

INTERPRETATION:
- System correctly identifies faces {metrics['accuracy']:.1%} of the time
- When system accepts a face, it's correct {metrics['precision']:.1%} of the time
- System catches {metrics['recall']:.1%} of actual positive cases
- Risk of accepting wrong person: {metrics['false_acceptance_rate']:.1%}
- Risk of rejecting correct person: {metrics['false_rejection_rate']:.1%}
"""
        return report
    
    def plot_performance_curves(self, lighting_condition=None, days_back=30):
        """Plot ROC curve and threshold analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT confidence_score, is_correct_identification
            FROM test_results 
            WHERE timestamp >= ?
        '''
        params = [datetime.now() - timedelta(days=days_back)]
        
        if lighting_condition:
            query += ' AND lighting_condition = ?'
            params.append(lighting_condition)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        # Calculate FAR and FRR for different thresholds
        min_conf = df['confidence_score'].min()
        max_conf = df['confidence_score'].max()
        thresholds = np.linspace(min_conf, max_conf, 50)
        
        far_rates = []
        frr_rates = []
        
        for threshold in thresholds:
            # Calculate metrics for this threshold
            false_accepts = len(df[(df['is_correct_identification'] == False) & 
                                 (df['confidence_score'] >= threshold)])
            total_negatives = len(df[df['is_correct_identification'] == False])
            far = false_accepts / total_negatives if total_negatives > 0 else 0
            
            false_rejects = len(df[(df['is_correct_identification'] == True) & 
                                 (df['confidence_score'] < threshold)])
            total_positives = len(df[df['is_correct_identification'] == True])
            frr = false_rejects / total_positives if total_positives > 0 else 0
            
            far_rates.append(far)
            frr_rates.append(frr)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: FAR vs FRR curves
        ax1.plot(thresholds, far_rates, label='False Acceptance Rate (FAR)', color='red')
        ax1.plot(thresholds, frr_rates, label='False Rejection Rate (FRR)', color='blue')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('FAR vs FRR Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Find and mark EER point
        differences = [abs(far - frr) for far, frr in zip(far_rates, frr_rates)]
        min_diff_idx = differences.index(min(differences))
        eer_threshold = thresholds[min_diff_idx]
        eer_rate = (far_rates[min_diff_idx] + frr_rates[min_diff_idx]) / 2
        ax1.plot(eer_threshold, eer_rate, 'go', markersize=10, label=f'EER: {eer_rate:.3f}')
        ax1.legend()
        
        # Plot 2: Confidence score distribution
        correct_scores = df[df['is_correct_identification'] == True]['confidence_score']
        incorrect_scores = df[df['is_correct_identification'] == False]['confidence_score']
        
        ax2.hist(correct_scores, bins=30, alpha=0.7, label='Correct Identifications', color='green')
        ax2.hist(incorrect_scores, bins=30, alpha=0.7, label='Incorrect Identifications', color='red')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def comparative_analysis_by_conditions(self, days_back=30):
        """Compare performance across different lighting conditions"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all lighting conditions
        conditions_query = '''
            SELECT DISTINCT lighting_condition 
            FROM test_results 
            WHERE timestamp >= ?
        '''
        conditions_df = pd.read_sql_query(conditions_query, conn, 
                                        params=[datetime.now() - timedelta(days=days_back)])
        
        results = {}
        
        for condition in conditions_df['lighting_condition']:
            metrics = self.calculate_metrics_from_db(condition, days_back)
            if metrics:
                results[condition] = metrics
        
        conn.close()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 
                          'false_acceptance_rate', 'false_rejection_rate', 'equal_error_rate']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Rate')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def real_time_metrics_tracker(self):
        """Track metrics in real-time during system operation"""
        class MetricsTracker:
            def __init__(self, analytics_instance):
                self.analytics = analytics_instance
                self.session_results = []
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            def add_result(self, actual_id, predicted_id, confidence, 
                          lighting_condition="normal", distance=2.0, face_angle="frontal"):
                """Add a result and update running metrics"""
                self.analytics.log_recognition_attempt(
                    actual_id, predicted_id, confidence, 
                    lighting_condition, distance, face_angle
                )
                
                # Calculate current session metrics
                current_metrics = self.analytics.calculate_metrics_from_db(days_back=1)
                
                if current_metrics:
                    print(f"Current Session Metrics:")
                    print(f"Accuracy: {current_metrics['accuracy']:.2%}")
                    print(f"Total Tests: {current_metrics['total_tests']}")
                    print("-" * 30)
            
            def get_session_summary(self):
                """Get summary of current session"""
                return self.analytics.calculate_metrics_from_db(days_back=1)
        
        return MetricsTracker(self)

# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize analytics system
#     analytics = FaceRecognitionAnalytics()
    
#     # Simulate some test data
#     print("Generating sample test data...")
    
#     # Simulate various scenarios
#     lighting_conditions = ['bright', 'normal', 'dim', 'very_dim']
#     students = ['STU001', 'STU002', 'STU003', 'STU004', 'STU005']
    
#     import random
    
#     for _ in range(200):  # 200 test cases
#         actual_id = random.choice(students)
#         lighting = random.choice(lighting_conditions)
        
#         # Simulate recognition results based on lighting
#         if lighting == 'bright':
#             confidence = random.uniform(20, 40)  # Good recognition
#             predicted_id = actual_id if random.random() > 0.05 else random.choice(students)
#         elif lighting == 'normal':
#             confidence = random.uniform(25, 50)
#             predicted_id = actual_id if random.random() > 0.10 else random.choice(students)
#         elif lighting == 'dim':
#             confidence = random.uniform(40, 70)
#             predicted_id = actual_id if random.random() > 0.20 else random.choice(students)
#         else:  # very_dim
#             confidence = random.uniform(60, 90)
#             predicted_id = actual_id if random.random() > 0.35 else random.choice(students)
        
#         analytics.log_recognition_attempt(
#             actual_id, predicted_id, confidence, lighting,
#             distance=random.uniform(1.5, 3.0),
#             face_angle=random.choice(['frontal', 'left', 'right'])
#         )
    
#     print("Sample data generated!")
    
#     # Generate performance report
#     print("\n" + "="*50)
#     print(analytics.generate_performance_report())
    
#     # Plot performance curves
#     analytics.plot_performance_curves()
    
#     # Comparative analysis
#     print("\nComparative Analysis by Lighting Conditions:")
#     comparison = analytics.comparative_analysis_by_conditions()
#     print(comparison)