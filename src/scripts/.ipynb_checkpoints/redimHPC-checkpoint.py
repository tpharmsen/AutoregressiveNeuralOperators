import os
import h5py
import torch
from pathlib import Path

# Parameters
data_path = 'data/prjs1359/PoolBoiling-SubCooled-FC72-2D-0.1/'
input_path = data_path #os.path.join(data_path, '')
output_path = os.path.join(data_path, 'redim_DS1/')
os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

keys_to_copy = [
    'real-runtime-params',
    'int-runtime-params'
]

TEMPERATURE = 'temperature'

print(f"Scaling temperature data in '{input_path}'\n and saving to '{output_path}'...")
# Process each file in the directory
files = [f for f in os.listdir(input_path) if f.endswith('.hdf5')]

for file in files:
    input_file = os.path.join(input_path, file)
    output_file = os.path.join(output_path, file)

    with h5py.File(input_file, 'r') as input_data:
        # Prepare output HDF5 file
        with h5py.File(output_file, 'w') as output_data:
            filestem = Path(file).stem
            wall_temp = None
            TWALL = 'Twall-'

            for key in input_data.keys():
                dataset = input_data[key][:]
                print(f"Processing key '{key}' in file '{file}' with shape {dataset.shape}")

                # Apply scaling only to the temperature dataset if filename contains 'Twall-'
                if key == TEMPERATURE and TWALL in filestem:
                    scaling_factor = int(filestem[len(TWALL):])  # Extract scaling factor from filename
                    temperature_data = torch.from_numpy(dataset)  # Convert to tensor
                    scaled_temperature = temperature_data * scaling_factor  # Apply scaling
                    scaled_temperature = scaled_temperature.numpy()  # Convert back to NumPy array
                    output_data.create_dataset(TEMPERATURE, data=scaled_temperature)  # Save scaled data
                    wall_temp = scaled_temperature.max()
                    print(f"Wall temperature scaling applied with factor {scaling_factor}. Max wall temp: {wall_temp}")

                else:
                    # Copy other datasets as-is
                    output_data.create_dataset(key, data=dataset)

            print(f"File '{file}' processed and saved to '{output_file}'.")

print("Scaling completed for all files.")
