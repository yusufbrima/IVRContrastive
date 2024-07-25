import splitfolders
from config import DATA_PATH, CHIMPANZEE_DATA_PATH

# Specify the path to your dataset
input_folder = DATA_PATH

# Specify the output folder where the split datasets will be saved
output_folder = CHIMPANZEE_DATA_PATH

# Split the dataset into training and testing using a 90-10 split
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.9, .1))

print("Dataset split into train and test sets successfully!")
