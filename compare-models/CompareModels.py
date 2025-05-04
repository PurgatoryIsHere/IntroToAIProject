import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import importlib.util
import os
import sys
import json
import tempfile
import nbformat
from nbconvert import PythonExporter


class ModelEvaluator:
    def __init__(self, data_path, models_dir):
        """
        Initialize the evaluator with the data path and directory containing model scripts
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        self.raw_data = None

    def load_data(self):
        """Load and prepare the dataset"""
        self.raw_data = pd.read_csv(self.data_path)
        if 'date' in self.raw_data.columns:
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'], errors='coerce')
            self.raw_data = self.raw_data.sort_values('date')
        print(f"Data loaded with {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
        return self.raw_data

    def convert_notebook_to_script(self, notebook_path):
        """Convert a Jupyter notebook to a Python script"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)

            # Convert notebook to Python script
            exporter = PythonExporter()
            python_script, _ = exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_content), as_version=4))

            # Remove IPython magic commands and display functions
            cleaned_script = ""
            for line in python_script.split('\n'):
                if not line.strip().startswith('%') and not line.strip().startswith('!') and 'display(' not in line:
                    cleaned_script += line + '\n'

            return cleaned_script
        except Exception as e:
            print(f"Error converting notebook to script: {str(e)}")
            return None

    def load_models(self):
        """Dynamically load all model scripts (.py files) and notebooks (.ipynb files) in the directory"""
        model_files = [f for f in os.listdir(self.models_dir)
                       if (f.endswith('.py') or f.endswith('.ipynb')) and f.startswith('Model')]

        for model_file in model_files:
            model_name = model_file.split('.')[0]
            file_ext = model_file.split('.')[-1]

            try:
                if file_ext == 'py':
                    # Load Python script
                    with open(os.path.join(self.models_dir, model_file), 'r', encoding='utf-8') as f:
                        code = f.read()
                elif file_ext == 'ipynb':
                    # Convert and load Jupyter notebook
                    code = self.convert_notebook_to_script(os.path.join(self.models_dir, model_file))
                    if code is None:
                        continue

                # Store the script content
                self.models[model_name] = {
                    'script': code,
                    'file': model_file,
                    'type': file_ext
                }
                print(f"Loaded model: {model_name} ({file_ext} file)")
            except Exception as e:
                print(f"Error loading {model_file}: {str(e)}")

        return self.models

    def extract_model_features(self, model_name):
        """Extract key features from a model script"""
        script = self.models[model_name]['script']
        features = {}

        # Detect model type
        if "LinearRegression" in script:
            features['model_type'] = "Linear Regression"
        elif "DecisionTreeRegressor" in script:
            features['model_type'] = "Decision Tree"
        elif "RandomForestRegressor" in script:
            features['model_type'] = "Random Forest"
        elif "XGBRegressor" in script:
            features['model_type'] = "XGBoost"
        elif "LSTM" in script or "Sequential" in script:
            features['model_type'] = "Neural Network/LSTM"
        else:
            features['model_type'] = "Unknown"

        # Detect features used
        features['features_used'] = []

        # Try to extract features list
        try:
            for line in script.split('\n'):
                if ("features = " in line or "X = " in line) and not line.strip().startswith('#'):
                    if "features = [" in line or "X = [" in line:
                        # Handle explicit list definition
                        feature_part = line.split('[', 1)[1].split(']')[0]
                        feature_list = feature_part.replace("'", "").replace('"', "").split(',')
                        features['features_used'] = [f.strip() for f in feature_list if f.strip()]
                        break
                    elif "features = " in line and ":" not in line:
                        # Handle other variable assignments
                        possible_list = line.split('=', 1)[1].strip()
                        if possible_list.startswith('[') and ']' in possible_list:
                            feature_list = possible_list.strip('[]').replace("'", "").replace('"', "").split(',')
                            features['features_used'] = [f.strip() for f in feature_list if f.strip()]
                            break
        except:
            pass

        if not features['features_used']:
            features['features_used'] = ["Could not extract features"]

        # Detect train-test split method
        if "train_test_split" in script:
            features['split_method'] = "Random"
        elif "iloc" in script and "split_idx" in script:
            features['split_method'] = "Chronological"
        else:
            features['split_method'] = "Unknown"

        return features

    def run_modified_model(self, model_name):
        """Run a modified version of the model that returns predictions and actual values"""
        script = self.models[model_name]['script']

        # Create a modified version for evaluation that captures test data and predictions
        eval_script = script.replace("plt.show()", "plt.close('all')")

        # Add code to save predictions and actual values
        eval_script += """
# Add to the end of the script
import pickle
result_data = {
    'y_test': y_test if 'y_test' in locals() else None,
    'y_pred': y_pred if 'y_pred' in locals() else None
}
with open('temp_model_results.pkl', 'wb') as f:
    pickle.dump(result_data, f)
"""

        # Create a unique temp file name to avoid conflicts when processing multiple models
        temp_file = f"temp_{model_name}_{os.getpid()}.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(eval_script)

        try:
            # Execute the modified script
            print(f"  Running {model_name}...")
            exit_code = os.system(f"python {temp_file}")

            if exit_code != 0:
                print(f"  Warning: Script exited with code {exit_code}")

            # Check if results file exists
            if not os.path.exists('temp_model_results.pkl'):
                print(f"  Error: Model did not generate results file.")
                return None

            # Load results
            with open('temp_model_results.pkl', 'rb') as f:
                import pickle
                results = pickle.load(f)

            # Verify results contain necessary data
            if results['y_test'] is None or results['y_pred'] is None:
                print(f"  Error: Model did not generate test data or predictions.")
                return None

            # Clean up
            os.remove(temp_file)
            os.remove('temp_model_results.pkl')

            return results
        except Exception as e:
            print(f"  Error running {model_name}: {str(e)}")
            # Clean up any temp files that may have been created
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists('temp_model_results.pkl'):
                os.remove('temp_model_results.pkl')
            return None

    def evaluate_all_models(self):
        """Evaluate all loaded models and collect metrics"""
        for model_name in self.models:
            print(f"\nEvaluating {model_name}...")

            # Extract model features
            model_features = self.extract_model_features(model_name)

            # Run the modified model
            results = self.run_modified_model(model_name)

            if results is not None:
                y_test = results['y_test']
                y_pred = results['y_pred']

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                try:
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                    accuracy = 100 - mape
                except:
                    mape = "N/A"
                    accuracy = r2 * 100  # Fallback to R2 for accuracy

                # Store results
                self.results[model_name] = {
                    **model_features,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape if isinstance(mape, str) else float(mape),
                    'Accuracy': accuracy,
                    'test_size': len(y_test),
                    'file_type': self.models[model_name]['type']
                }

                print(f"  Model type: {model_features['model_type']}")
                print(
                    f"  Features: {', '.join(model_features['features_used'][:3])}{'...' if len(model_features['features_used']) > 3 else ''}")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
                print(f"  Accuracy: {accuracy:.2f}%")
            else:
                print(f"  Failed to evaluate")

        return self.results

    def compare_models(self):
        """Create comparison visualizations of all models"""
        if not self.results:
            print("No results to compare. Run evaluate_all_models first.")
            return

        # Create DataFrame of results
        results_df = pd.DataFrame.from_dict(self.results, orient='index')

        # Handle non-numeric MAPE values
        if 'MAPE' in results_df.columns:
            results_df['MAPE'] = pd.to_numeric(results_df['MAPE'], errors='coerce')

        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        results_df['Accuracy'].plot(kind='bar', color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png')

        # Error metrics comparison
        plt.figure(figsize=(14, 8))
        metrics = ['MAE', 'RMSE']
        results_df[metrics].plot(kind='bar')
        plt.title('Error Metrics Comparison')
        plt.ylabel('Error Value')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_error_comparison.png')

        # R² comparison
        plt.figure(figsize=(12, 6))
        results_df['R2'].plot(kind='bar', color='green')
        plt.title('R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_r2_comparison.png')

        # Export results to CSV
        results_df.to_csv('model_comparison_results.csv')
        print(f"Comparison complete. Results saved to CSV and visualizations generated.")

        return results_df

    def generate_report(self):
        """Generate a comprehensive HTML report of the model comparison"""
        if not self.results:
            print("No results to report. Run evaluate_all_models first.")
            return

        results_df = pd.DataFrame.from_dict(self.results, orient='index')

        # Create HTML report
        html = """
        <html>
        <head>
            <title>Stock Prediction Models Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .model-details { margin-top: 30px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
                .metric-good { color: green; font-weight: bold; }
                .metric-bad { color: red; }
                h1, h2 { color: #333366; }
                .file-py { color: blue; }
                .file-ipynb { color: orange; }
            </style>
        </head>
        <body>
            <h1>Stock Prediction Models Comparison Report</h1>

            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>File Type</th>
                    <th>Accuracy (%)</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                    <th>Split Method</th>
                </tr>
        """

        # Add rows for each model
        for model_name, results in self.results.items():
            accuracy_class = "metric-good" if results['Accuracy'] > 90 else "metric-bad"
            r2_class = "metric-good" if results['R2'] > 0.8 else "metric-bad"
            file_class = f"file-{results.get('file_type', 'py')}"

            html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{results['model_type']}</td>
                    <td class="{file_class}">{results.get('file_type', 'py')}</td>
                    <td class="{accuracy_class}">{results['Accuracy']:.2f}%</td>
                    <td>{results['MAE']:.4f}</td>
                    <td>{results['RMSE']:.4f}</td>
                    <td class="{r2_class}">{results['R2']:.4f}</td>
                    <td>{results['split_method']}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Model Details</h2>
        """

        # Add details for each model
        for model_name, model_info in self.models.items():
            if model_name not in self.results:
                continue

            features_list = ', '.join(self.results[model_name]['features_used'])

            html += f"""
            <div class="model-details">
                <h3>{model_name} ({model_info['type']} file)</h3>
                <p><strong>Model Type:</strong> {self.results[model_name]['model_type']}</p>
                <p><strong>Features Used:</strong> {features_list}</p>
                <p><strong>Split Method:</strong> {self.results[model_name]['split_method']}</p>
                <p><strong>Test Size:</strong> {self.results[model_name]['test_size']} samples</p>
                <p><strong>Performance Metrics:</strong></p>
                <ul>
                    <li>Accuracy: {self.results[model_name]['Accuracy']:.2f}%</li>
                    <li>MAE: {self.results[model_name]['MAE']:.4f}</li>
                    <li>MSE: {self.results[model_name]['MSE']:.4f}</li>
                    <li>RMSE: {self.results[model_name]['RMSE']:.4f}</li>
                    <li>R²: {self.results[model_name]['R2']:.4f}</li>
                </ul>
            </div>
            """

        html += """
            <h2>Conclusions</h2>
            <p>Based on the metrics above, the best performing model is: 
        """

        # Find best model based on accuracy
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['Accuracy'])
            html += f"<strong>{best_model[0]}</strong> with an accuracy of {best_model[1]['Accuracy']:.2f}%.</p>"

            html += f"""
                <p>Best model by different metrics:</p>
                <ul>
                    <li><strong>Lowest MAE:</strong> {min(self.results.items(), key=lambda x: x[1]['MAE'])[0]}</li>
                    <li><strong>Lowest RMSE:</strong> {min(self.results.items(), key=lambda x: x[1]['RMSE'])[0]}</li>
                    <li><strong>Highest R²:</strong> {max(self.results.items(), key=lambda x: x[1]['R2'])[0]}</li>
                </ul>
            """

        html += """
            <p>Recommendations:</p>
            <ul>
                <li>Consider ensemble methods combining the strengths of different models</li>
                <li>Evaluate models on different time periods to test robustness</li>
                <li>Explore feature importance to understand key drivers</li>
            </ul>

            <p>Images:</p>
            <img src="model_accuracy_comparison.png" alt="Accuracy Comparison" style="max-width: 100%;">
            <img src="model_error_comparison.png" alt="Error Metrics Comparison" style="max-width: 100%;">
            <img src="model_r2_comparison.png" alt="R² Score Comparison" style="max-width: 100%;">
        </body>
        </html>
        """

        # Write HTML report to file
        with open('model_comparison_report.html', 'w') as f:
            f.write(html)

        print("Detailed HTML report generated: model_comparison_report.html")


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator(data_path='../DATA/JPStockPredict.csv', models_dir='../models')
    evaluator.load_data()
    evaluator.load_models()
    evaluator.evaluate_all_models()
    results = evaluator.compare_models()
    evaluator.generate_report()

    print("\nSummary of model comparison:")
    for model_name, result in evaluator.results.items():
        print(f"{model_name}: Accuracy = {result['Accuracy']:.2f}%, R² = {result['R2']:.4f}")