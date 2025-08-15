#!/usr/bin/env python
"""
ClearML credentials setup script
"""
import os
import sys

def setup_credentials():
    print("ClearML Credentials Setup")
    print("-----------------------\n")
    print("Please enter the credentials from your ClearML profile settings:")
    
    access_key = input("Access Key: ").strip()
    secret_key = input("Secret Key: ").strip()
    
    if not access_key or not secret_key:
        print("Error: Access Key and Secret Key cannot be empty.")
        sys.exit(1)
    
    config_dir = os.path.expanduser("~/.clearml")
    os.makedirs(config_dir, exist_ok=True)
    
    config_content = f"""api {{
    web_server: "http://localhost:8080",
    api_server: "http://localhost:8008", 
    files_server: "http://localhost:8081",
    credentials {{
        "access_key" = "{access_key}"
        "secret_key" = "{secret_key}"
    }}
}}

sdk {{
    metrics {{
        # History size for matplotlib (in images)
        matplotlib_untitled_history_size: 100
        # Log text plots as svg files
        svg_fonts: true
        # Max number of files (images) stored per debug sample
        file_history_size: 100
        # Max history size for Tensorboard events
        tensorboard_single_series_per_graph: false
    }}
}}
"""
    
    config_file = os.path.join(config_dir, "clearml.conf")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Credentials saved to {config_file}")
    print("✅ ClearML configuration completed!")

if __name__ == "__main__":
    setup_credentials()
