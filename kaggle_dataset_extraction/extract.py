import kagglehub

# Download latest version
path = kagglehub.dataset_download("sebastianwillmann/beverage-sales")

print("Path to dataset files:", path) 