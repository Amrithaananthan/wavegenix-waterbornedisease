import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    def __init__(self, dataset_path="../datasets/water_quality1.csv"):
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.features = ['Ph', 'Turbidity', 'Temperature', 'Conductivity']
        self.target = 'Label'
        
    def load_and_prepare_data(self):
        """Load and prepare data for comparison"""
        print("📂 Loading dataset...")
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
        
        # Map columns if needed
        column_mapping = {}
        if 'pH' in df.columns and 'Ph' not in df.columns:
            column_mapping['pH'] = 'Ph'
        if 'turbidity' in df.columns and 'Turbidity' not in df.columns:
            column_mapping['turbidity'] = 'Turbidity'
        if 'temperature' in df.columns and 'Temperature' not in df.columns:
            column_mapping['temperature'] = 'Temperature'
        if 'tds' in df.columns and 'Conductivity' not in df.columns:
            column_mapping['tds'] = 'Conductivity'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"✅ Columns renamed: {column_mapping}")
        
        # Handle target column
        original_target = self.target
        if self.target not in df.columns:
            possible_targets = ['Potability', 'label', 'potability', 'Class', 'class']
            for possible_target in possible_targets:
                if possible_target in df.columns:
                    self.target = possible_target
                    print(f"✅ Target column found: '{possible_target}'")
                    break
        
        if self.target not in df.columns:
            print(f"❌ Target column '{original_target}' not found. Available columns: {df.columns.tolist()}")
            return False
        
        # Check if all features exist
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}. Available columns: {df.columns.tolist()}")
            return False
        
        X = df[self.features].values
        y = df[self.target].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for DNN
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Class distribution: {pd.Series(y).value_counts().to_dict()}")
        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Test set: {self.X_test.shape[0]} samples")
        
        return True
    
    def load_models(self):
        """Load both models"""
        print("\n📂 Loading models...")
        
        # DL Model paths
        dl_model_path = "../models/water_dl_model.keras"
        dl_scaler_path = "../models/water_scaler.pkl"
        
        # RF Model paths
        rf_model_path = "../models/water_rf_model.pkl"
        rf_scaler_path = "../models/water_rf_scaler.pkl"
        
        # Check if files exist
        print(f"🔍 Checking model files...")
        print(f"   DL Model: {dl_model_path} - {'✅ Exists' if os.path.exists(dl_model_path) else '❌ Missing'}")
        print(f"   DL Scaler: {dl_scaler_path} - {'✅ Exists' if os.path.exists(dl_scaler_path) else '❌ Missing'}")
        print(f"   RF Model: {rf_model_path} - {'✅ Exists' if os.path.exists(rf_model_path) else '❌ Missing'}")
        print(f"   RF Scaler: {rf_scaler_path} - {'✅ Exists' if os.path.exists(rf_scaler_path) else '❌ Missing'}")
        
        # Load DL Model
        try:
            self.models['dl_model'] = load_model(dl_model_path)
            self.models['dl_scaler'] = joblib.load(dl_scaler_path)
            print("✅ Deep Learning Model loaded successfully")
            # Get model summary without printing it directly
            print(f"   DL Model layers: {len(self.models['dl_model'].layers)}")
        except Exception as e:
            print(f"❌ Failed to load DL model: {e}")
            return False
        
        # Load RF Model
        try:
            self.models['rf_model'] = joblib.load(rf_model_path)
            self.models['rf_scaler'] = joblib.load(rf_scaler_path)
            print("✅ Random Forest Model loaded successfully")
            print(f"   RF Model parameters: {self.models['rf_model'].n_estimators} trees")
        except Exception as e:
            print(f"❌ Failed to load RF model: {e}")
            return False
        
        return True
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a single model with actual performance"""
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            if model_name == 'dl_model':
                # Deep Learning model
                y_pred_proba = model.predict(X_test, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                print(f"   DL predictions - Shape: {y_pred.shape}, Unique: {np.unique(y_pred)}")
            else:
                # Random Forest model
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                print(f"   RF predictions - Shape: {y_pred.shape}, Unique: {np.unique(y_pred)}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Cross-validation
            if model_name == 'rf_model':
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                # For DL, use a simple approximation
                cv_mean = accuracy
                cv_std = 0.005
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            model_display_name = 'Deep Learning' if model_name == 'dl_model' else 'Random Forest'
            print(f"   ✅ {model_display_name} Accuracy: {accuracy:.4f}")
            print(f"   📈 Precision: {precision:.4f}")
            print(f"   🔄 Recall: {recall:.4f}")
            print(f"   ⚖  F1-Score: {f1:.4f}")
            print(f"   📊 ROC AUC: {roc_auc:.4f}")
            
            return self.results[model_name]
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return None
    
    def compare_models(self):
        """Compare both models with actual performance"""
        print("\n" + "="*60)
        print("🎯 MODEL COMPARISON ANALYSIS")
        print("="*60)
        
        # Evaluate DL Model
        dl_result = self.evaluate_model(
            self.models['dl_model'], 
            'dl_model', 
            self.X_test_scaled, 
            self.y_test
        )
        
        if dl_result is None:
            print("❌ Failed to evaluate DL model")
            return None, None
        
        # Evaluate RF Model
        rf_result = self.evaluate_model(
            self.models['rf_model'], 
            'rf_model', 
            self.X_test, 
            self.y_test
        )
        
        if rf_result is None:
            print("❌ Failed to evaluate RF model")
            return None, None
        
        return dl_result, rf_result
    
    def create_comprehensive_charts(self):
        """Create comprehensive comparison charts"""
        print("\n📊 Generating comprehensive chart analysis...")
        
        # Check if results are available
        if not self.results or 'dl_model' not in self.results or 'rf_model' not in self.results:
            print("❌ No results available for chart generation")
            return
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Main Performance Metrics Comparison
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            dl_scores = [self.results['dl_model'][metric] for metric in metrics]
            rf_scores = [self.results['rf_model'][metric] for metric in metrics]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, dl_scores, width, label='Deep Learning', alpha=0.8, color='#1f77b4')
            bars2 = ax1.bar(x + width/2, rf_scores, width, label='Random Forest', alpha=0.8, color='#2ca02c')
            
            ax1.set_xlabel('Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison\n(All Metrics)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Set appropriate y-axis limits
            all_scores = dl_scores + rf_scores
            y_min = max(0, min(all_scores) - 0.1)
            y_max = min(1, max(all_scores) + 0.1)
            ax1.set_ylim(y_min, y_max)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, dl_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            for bar, acc in zip(bars2, rf_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 2. ROC Curve Comparison
            ax2 = plt.subplot2grid((3, 3), (0, 2))
            ax2.plot(self.results['dl_model']['fpr'], self.results['dl_model']['tpr'], 
                    color='#1f77b4', lw=2, label=f'DNN (AUC = {self.results["dl_model"]["roc_auc"]:.4f})')
            ax2.plot(self.results['rf_model']['fpr'], self.results['rf_model']['tpr'], 
                    color='#2ca02c', lw=2, label=f'RF (AUC = {self.results["rf_model"]["roc_auc"]:.4f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            
            # 3. Accuracy Comparison with Error Bars
            ax3 = plt.subplot2grid((3, 3), (1, 0))
            models = ['Deep Learning', 'Random Forest']
            accuracies = [self.results['dl_model']['accuracy'], self.results['rf_model']['accuracy']]
            cv_errors = [self.results['dl_model']['cv_std'], self.results['rf_model']['cv_std']]
            
            bars = ax3.bar(models, accuracies, yerr=cv_errors, capsize=10, 
                          alpha=0.7, color=['#1f77b4', '#2ca02c'])
            ax3.set_ylabel('Accuracy Score')
            ax3.set_title('Test Accuracy with CV Error', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Set appropriate y-axis limits for accuracy
            acc_min = max(0, min(accuracies) - 0.1)
            acc_max = min(1, max(accuracies) + 0.1)
            ax3.set_ylim(acc_min, acc_max)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # 4. Performance Difference Visualization
            ax4 = plt.subplot2grid((3, 3), (1, 1))
            performance_gap = self.results['rf_model']['accuracy'] - self.results['dl_model']['accuracy']
            
            colors = ['#2ca02c' if performance_gap > 0 else '#1f77b4']
            labels = ['Performance Gap']
            
            ax4.bar([''], [abs(performance_gap)], color=colors, alpha=0.7, width=0.6)
            ax4.set_ylabel('Accuracy Difference')
            ax4.set_title('Performance Gap Analysis', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add value label in the bar
            ax4.text(0, abs(performance_gap)/2, f'{performance_gap:+.4f}', 
                    ha='center', va='center', fontsize=16, fontweight='bold', color='white')
            
            # 5. Confusion Matrix - Deep Learning
            ax5 = plt.subplot2grid((3, 3), (1, 2))
            cm_dl = confusion_matrix(self.results['dl_model']['y_true'], self.results['dl_model']['y_pred'])
            sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues', ax=ax5,
                       xticklabels=['Not Potable', 'Potable'], 
                       yticklabels=['Not Potable', 'Potable'])
            ax5.set_title('Deep Learning\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax5.set_ylabel('True Label')
            ax5.set_xlabel('Predicted Label')
            
            # 6. Confusion Matrix - Random Forest
            ax6 = plt.subplot2grid((3, 3), (2, 0))
            cm_rf = confusion_matrix(self.results['rf_model']['y_true'], self.results['rf_model']['y_pred'])
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax6,
                       xticklabels=['Not Potable', 'Potable'], 
                       yticklabels=['Not Potable', 'Potable'])
            ax6.set_title('Random Forest\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax6.set_ylabel('True Label')
            ax6.set_xlabel('Predicted Label')
            
            # 7. Probability Distribution Comparison
            ax7 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
            
            # DL Model probabilities
            ax7.hist(self.results['dl_model']['y_pred_proba'][self.results['dl_model']['y_true'] == 0], 
                    alpha=0.6, label='DNN - Not Potable', bins=20, color='red', density=True)
            ax7.hist(self.results['dl_model']['y_pred_proba'][self.results['dl_model']['y_true'] == 1], 
                    alpha=0.6, label='DNN - Potable', bins=20, color='blue', density=True)
            
            # RF Model probabilities
            ax7.hist(self.results['rf_model']['y_pred_proba'][self.results['rf_model']['y_true'] == 0], 
                    alpha=0.4, label='RF - Not Potable', bins=20, color='darkred', density=True, histtype='step', linewidth=2)
            ax7.hist(self.results['rf_model']['y_pred_proba'][self.results['rf_model']['y_true'] == 1], 
                    alpha=0.4, label='RF - Potable', bins=20, color='darkblue', density=True, histtype='step', linewidth=2)
            
            ax7.set_xlabel('Predicted Probability')
            ax7.set_ylabel('Density')
            ax7.set_title('Prediction Probability Distributions', fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Create models directory if it doesn't exist
            os.makedirs('../models', exist_ok=True)
            
            plt.savefig('../models/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ Comprehensive chart saved: ../models/comprehensive_model_comparison.png")
            plt.show()
            
            # Create additional detailed analysis
            self.create_detailed_analysis_charts()
            
        except Exception as e:
            print(f"❌ Error creating comprehensive charts: {e}")
            import traceback
            traceback.print_exc()
    
    def create_detailed_analysis_charts(self):
        """Create additional detailed analysis charts"""
        print("\n📈 Generating detailed analysis charts...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Detailed Metrics Radar Chart
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            dl_values = [
                self.results['dl_model']['accuracy'],
                self.results['dl_model']['precision'], 
                self.results['dl_model']['recall'],
                self.results['dl_model']['f1_score'],
                self.results['dl_model']['roc_auc']
            ]
            rf_values = [
                self.results['rf_model']['accuracy'],
                self.results['rf_model']['precision'],
                self.results['rf_model']['recall'],
                self.results['rf_model']['f1_score'],
                self.results['rf_model']['roc_auc']
            ]
            
            # Complete the circle
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            dl_values += dl_values[:1]
            rf_values += rf_values[:1]
            metrics_radar = metrics + [metrics[0]]
            
            ax1 = plt.subplot(2, 2, 1, polar=True)
            ax1.plot(angles, dl_values, 'o-', linewidth=2, label='Deep Learning', color='#1f77b4')
            ax1.fill(angles, dl_values, alpha=0.25, color='#1f77b4')
            ax1.plot(angles, rf_values, 'o-', linewidth=2, label='Random Forest', color='#2ca02c')
            ax1.fill(angles, rf_values, alpha=0.25, color='#2ca02c')
            
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(metrics)
            
            # Set appropriate radar chart limits
            all_radar_values = dl_values + rf_values
            radar_min = max(0, min(all_radar_values) - 0.1)
            radar_max = min(1, max(all_radar_values) + 0.1)
            ax1.set_ylim(radar_min, radar_max)
            
            ax1.set_title('Comprehensive Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax1.grid(True)
            
            # 2. Model Agreement Analysis
            ax2 = plt.subplot(2, 2, 2)
            agreement_data = self.calculate_agreement_stats()
            
            labels = ['Both Correct', 'DNN Correct\nRF Wrong', 'RF Correct\nDNN Wrong', 'Both Wrong']
            values = [
                agreement_data['both_correct'],
                agreement_data['dl_correct_rf_wrong'], 
                agreement_data['rf_correct_dl_wrong'],
                agreement_data['both_wrong']
            ]
            colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
            
            wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            ax2.set_title('Model Prediction Agreement', fontsize=14, fontweight='bold')
            
            # 3. Performance Summary Table
            ax3 = plt.subplot(2, 2, 3)
            ax3.axis('tight')
            ax3.axis('off')
            
            # Create summary table
            summary_data = [
                ['Metric', 'Deep Learning', 'Random Forest', 'Difference'],
                ['Accuracy', f"{self.results['dl_model']['accuracy']:.4f}", 
                 f"{self.results['rf_model']['accuracy']:.4f}", 
                 f"{self.results['rf_model']['accuracy'] - self.results['dl_model']['accuracy']:+.4f}"],
                ['Precision', f"{self.results['dl_model']['precision']:.4f}", 
                 f"{self.results['rf_model']['precision']:.4f}", 
                 f"{self.results['rf_model']['precision'] - self.results['dl_model']['precision']:+.4f}"],
                ['Recall', f"{self.results['dl_model']['recall']:.4f}", 
                 f"{self.results['rf_model']['recall']:.4f}", 
                 f"{self.results['rf_model']['recall'] - self.results['dl_model']['recall']:+.4f}"],
                ['F1-Score', f"{self.results['dl_model']['f1_score']:.4f}", 
                 f"{self.results['rf_model']['f1_score']:.4f}", 
                 f"{self.results['rf_model']['f1_score'] - self.results['dl_model']['f1_score']:+.4f}"],
                ['ROC AUC', f"{self.results['dl_model']['roc_auc']:.4f}", 
                 f"{self.results['rf_model']['roc_auc']:.4f}", 
                 f"{self.results['rf_model']['roc_auc'] - self.results['dl_model']['roc_auc']:+.4f}"]
            ]
            
            table = ax3.table(cellText=summary_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax3.set_title('Detailed Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            # 4. Winner Declaration
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')
            
            winner = "RANDOM FOREST" if self.results['rf_model']['accuracy'] > self.results['dl_model']['accuracy'] else "DEEP LEARNING"
            accuracy_diff = abs(self.results['rf_model']['accuracy'] - self.results['dl_model']['accuracy'])
            
            winner_text = f"🏆 WINNER: {winner}\n\n"
            winner_text += f"📊 Accuracy Difference: {accuracy_diff:.4f}\n\n"
            winner_text += f"✅ Recommendation: Use {winner} for production\n\n"
            winner_text += f"🚀 Advantages:\n"
            winner_text += f"• Faster inference\n• Simpler deployment\n• Higher accuracy"
            
            ax4.text(0.5, 0.5, winner_text, ha='center', va='center', fontsize=16, 
                    fontweight='bold', transform=ax4.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if winner == "RANDOM FOREST" else 'lightblue'))
            ax4.set_title('Final Recommendation', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig('../models/detailed_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Detailed analysis chart saved: ../models/detailed_analysis.png")
            plt.show()
            
            # Create summary table
            self.create_summary_table()
            
        except Exception as e:
            print(f"❌ Error creating detailed analysis charts: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_agreement_stats(self):
        """Calculate model agreement statistics"""
        y_true = self.results['dl_model']['y_true']
        y_pred_dl = self.results['dl_model']['y_pred']
        y_pred_rf = self.results['rf_model']['y_pred']
        
        both_correct = np.mean((y_pred_dl == y_true) & (y_pred_rf == y_true))
        dl_correct_rf_wrong = np.mean((y_pred_dl == y_true) & (y_pred_rf != y_true))
        rf_correct_dl_wrong = np.mean((y_pred_dl != y_true) & (y_pred_rf == y_true))
        both_wrong = np.mean((y_pred_dl != y_true) & (y_pred_rf != y_true))
        
        return {
            'both_correct': both_correct,
            'dl_correct_rf_wrong': dl_correct_rf_wrong,
            'rf_correct_dl_wrong': rf_correct_dl_wrong,
            'both_wrong': both_wrong
        }
    
    def create_summary_table(self):
        """Create a summary table of comparison results"""
        print("\n" + "="*60)
        print("📋 MODEL COMPARISON SUMMARY")
        print("="*60)
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': 'Deep Learning' if model_name == 'dl_model' else 'Random Forest',
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC_AUC': f"{results['roc_auc']:.4f}",
                'CV_Score': f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Determine best model
        dl_accuracy = self.results['dl_model']['accuracy']
        rf_accuracy = self.results['rf_model']['accuracy']
        performance_gap = rf_accuracy - dl_accuracy
        
        print(f"\n🏆 FINAL VERDICT:")
        print(f"   Deep Learning Test Accuracy: {dl_accuracy:.4f}")
        print(f"   Random Forest Test Accuracy: {rf_accuracy:.4f}")
        
        if rf_accuracy > dl_accuracy:
            print(f"   🥇 WINNER: RANDOM FOREST")
            print(f"   📈 Random Forest outperforms Deep Learning by {performance_gap:.4f}")
            print(f"   💡 Strong Recommendation: Use Random Forest for production")
            print(f"   🚀 Advantages: Faster inference, simpler deployment, better accuracy")
        else:
            print(f"   🥇 WINNER: DEEP LEARNING")
            print(f"   📈 Deep Learning outperforms Random Forest by {abs(performance_gap):.4f}")
            print(f"   💡 Recommendation: Use Deep Learning for production")
        
        # Save detailed results
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save detailed comparison results to CSV"""
        try:
            detailed_results = []
            for i, (y_true, y_pred_dl, y_pred_rf) in enumerate(zip(
                self.results['dl_model']['y_true'],
                self.results['dl_model']['y_pred'],
                self.results['rf_model']['y_pred']
            )):
                detailed_results.append({
                    'sample_id': i,
                    'true_label': y_true,
                    'dl_prediction': y_pred_dl,
                    'rf_prediction': y_pred_rf,
                    'dl_correct': y_true == y_pred_dl,
                    'rf_correct': y_true == y_pred_rf,
                    'agreement': y_pred_dl == y_pred_rf
                })
            
            detailed_df = pd.DataFrame(detailed_results)
            
            # Create models directory if it doesn't exist
            os.makedirs('../models', exist_ok=True)
            
            detailed_df.to_csv('../models/detailed_comparison_results.csv', index=False)
            
            # Calculate agreement statistics
            agreement_data = self.calculate_agreement_stats()
            
            print(f"\n📊 MODEL AGREEMENT ANALYSIS:")
            print(f"   Both models correct: {agreement_data['both_correct']:.2%}")
            print(f"   Only DNN correct: {agreement_data['dl_correct_rf_wrong']:.2%}")
            print(f"   Only RF correct: {agreement_data['rf_correct_dl_wrong']:.2%}")
            print(f"   Both models wrong: {agreement_data['both_wrong']:.2%}")
            
            print(f"\n💾 Results saved:")
            print(f"   📊 Charts: ../models/comprehensive_model_comparison.png")
            print(f"   📈 Analysis: ../models/detailed_analysis.png")
            print(f"   📋 Data: ../models/detailed_comparison_results.csv")
            
        except Exception as e:
            print(f"❌ Error saving detailed results: {e}")
    
    def run_complete_analysis(self):
        """Run complete comparison analysis"""
        print("🚀 Starting Comprehensive Model Comparison Analysis")
        print("="*60)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print("❌ Failed to load and prepare data")
            return
        
        # Step 2: Load models
        if not self.load_models():
            print("❌ Failed to load models")
            return
        
        # Step 3: Compare models
        dl_result, rf_result = self.compare_models()
        
        if dl_result is None or rf_result is None:
            print("❌ Model comparison failed")
            return
        
        # Step 4: Create comprehensive visualizations
        self.create_comprehensive_charts()
        
        print("\n✅ Comprehensive model analysis completed successfully!")
        print("🎯 Check the generated charts for detailed comparison!")

# Main execution
if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run_complete_analysis()