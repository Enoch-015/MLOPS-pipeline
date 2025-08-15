#!/usr/bin/env python
"""
ClearML basic test script - this will create a ClearML Task
which will appear in your ClearML Web UI.
"""
from clearml import Task

def main():
    # Initialize a new task
    task = Task.init(
        project_name="ClearML Test",
        task_name="Connection Test",
        output_uri=True
    )
    
    # Add some parameters
    params = {
        "test_param": 42,
        "test_string": "Hello ClearML!",
        "test_list": [1, 2, 3]
    }
    task.connect(params)
    
    # Log some scalars
    for i in range(10):
        task.get_logger().report_scalar(
            title="Test Metrics", 
            series="accuracy", 
            value=i*0.1, 
            iteration=i
        )
    
    print("✅ ClearML task created successfully!")
    print(f"✅ Task ID: {task.id}")
    print("✅ Check the task in the ClearML UI: http://localhost:8080")

if __name__ == "__main__":
    main()
