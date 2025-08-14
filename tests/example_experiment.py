"""
Example ClearML MLOps Script
This demonstrates basic ClearML integration for experiment tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ClearML package for experiment tracking
from clearml import Task

def run_ml_experiment():
    """Run a simple ML experiment with ClearML tracking"""
    
    # Initialize ClearML Task
    task = Task.init(
        project_name="MLOps Pipeline Demo",
        task_name="Random Forest Classification",
        tags=["sklearn", "classification", "demo"]
    )
    
    # Hyperparameters
    config = {
        'n_samples': 1000,
        'n_features': 20,
        'n_informative': 15,
        'n_redundant': 5,
        'random_state': 42,
        'test_size': 0.3,
        'n_estimators': 100,
        'max_depth': 10
    }
    
    # Log hyperparameters
    task.connect(config)
    
    print("🔬 Generating synthetic dataset...")
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        n_informative=config['n_informative'],
        n_redundant=config['n_redundant'],
        random_state=config['random_state']
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🏋️ Training samples: {X_train.shape[0]}")
    print(f"🧪 Test samples: {X_test.shape[0]}")
    
    # Train model
    print("🚂 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=config['random_state']
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    # Log metrics
    task.get_logger().report_scalar("Performance", "Accuracy", accuracy, iteration=0)
    
    # Create and log a simple plot
    feature_importance = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Log plot
    task.get_logger().report_matplotlib_figure("Feature Importance", "importance", plt.gcf())
    
    plt.savefig('feature_importance.png')
    plt.show()
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n✅ Experiment completed successfully!")
    print("🌐 Check the ClearML web interface at http://localhost:8080 to view results")
    
    return model, accuracy

if __name__ == "__main__":
    print("🚀 Starting MLOps Pipeline Demo")
    print("=" * 50)
    
    # Check if ClearML server is running
    try:
        import requests
        response = requests.get("http://localhost:8008/debug.ping", timeout=5)
        if response.status_code == 200:
            print("✅ ClearML server is running")
        else:
            print("⚠️ ClearML server might not be fully ready")
    except:
        print("❌ ClearML server is not accessible")
        print("💡 Start it with: docker-compose up -d")
    
    print("=" * 50)
    
    model, accuracy = run_ml_experiment()
    
    print("=" * 50)
    print("🎉 Demo completed!")
    print(f"📈 Final accuracy: {accuracy:.4f}")
    print("🔗 Next steps:")
    print("  1. Install clearml: pip install clearml")
    print("  2. Configure credentials: clearml-init")
    print("  3. Uncomment ClearML lines in this script")
    print("  4. Re-run to see full tracking capabilities")
