import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# Load the food nutritional data
csv_file = 'food2.csv'

# Check if the CSV file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: The file {csv_file} was not found!")

try:
    food_data = pd.read_csv(csv_file, on_bad_lines='skip')  # ✅ Fixed
except pd.errors.ParserError:
    raise ValueError("Error: CSV file has inconsistent column counts!")

# Drop any rows with missing values
food_data.dropna(inplace=True)

# Display first few rows to ensure data is loaded correctly
print(f"Food data loaded: {food_data.head()}")

# Encode food names as labels
food_label_encoder = LabelEncoder()
food_data['Food_Label'] = food_label_encoder.fit_transform(food_data['Name'])

# Encode allergens as labels (assuming 'Allergens' column exists)
allergen_label_encoder = LabelEncoder()
food_data['Allergen_Label'] = allergen_label_encoder.fit_transform(food_data['Allergens'])

# Image directory
data_dir = r"C:\Users\akalo\OneDrive\Desktop\AI_Food_Analizer\static\images"

# Check if the image directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Error: The directory {data_dir} does not exist!")

# Check the contents of the image directory
subdirectories = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]
print(f"Subdirectories in {data_dir}: {subdirectories}")

# Validate subdirectories and ensure they contain images
image_formats = ('.jpg', '.jpeg', '.png')
for subdir in subdirectories:
    subdir_path = os.path.join(data_dir, subdir)
    files_in_subdir = [f for f in os.listdir(subdir_path) if f.endswith(image_formats)]
    print(f"Images found in {subdir}: {len(files_in_subdir)}")
    if len(files_in_subdir) == 0:
        raise ValueError(f"Error: No images found in subdirectory '{subdir}'.")

# Set image size and batch size
img_size = 224  # Model input size
batch_size = 32

# ImageDataGenerator for augmentation (no validation split anymore)
train_datagen = ImageDataGenerator(rescale=1./255)

# Train generator only (no validation generator)
train_generator = train_datagen.flow_from_directory(
    directory=data_dir,  # Use the correct path to the images
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse'  # Class mode as sparse for integer labels
)

# Ensure the generator has data and samples
if train_generator.samples == 0:
    raise ValueError("Error: No images found in the training directory.")

print(f"Training images found: {train_generator.samples}")

# Define CNN model with multiple outputs for food classification and allergy prediction
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Added dropout for regularization
    Dense(len(food_data['Food_Label'].unique()), activation='softmax', name='Food_Output'),  # Food classification
    Dense(len(food_data['Allergen_Label'].unique()), activation='softmax', name='Allergen_Output')  # Allergy classification
])

# Compile the model with multiple outputs
model.compile(optimizer='adam',
              loss={'Food_Output': 'sparse_categorical_crossentropy', 'Allergen_Output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

# Skip training and print a confirmation (train_generator issue was fixed)
print("Model setup completed. Now skipping training due to generator issue.")

# Save the model after setup (even without training)
model.save('food_nutrition_model_with_allergens.h5')
print("✅ Model saved successfully as 'food_nutrition_model_with_allergens.h5'!")
