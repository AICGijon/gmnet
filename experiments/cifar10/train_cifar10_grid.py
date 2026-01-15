import argparse
import json
import os
import subprocess
import time

def run_experiment_process(base_params_path, n_gm_layers, gaussian_value, cuda_device):
    """
    Prepares the parameters, saves them to a temporary JSON file, and launches
    the train_cifar10.py script as a subprocess.
    """
    # Load base parameters from the provided path
    with open(base_params_path, 'r') as f:
        params = json.load(f)

    # Update n_gm_layers parameter
    params["n_gm_layers"] = n_gm_layers

    # Update num_gaussians: A list of 'gaussian_value' repeated 'n_gm_layers' times
    params["num_gaussians"] = [gaussian_value] * n_gm_layers

    # Update gaussian_dimensions: Assuming it's a list of fixed value '5' repeated 'n_gm_layers' times.
    # This aligns with the structure observed in 'gmnet_reg01.json'.
    params["gaussian_dimensions"] = [5] * n_gm_layers

    if n_gm_layers == 1:
        params['cka_regularization'] = 0

    # Construct a unique training name for this experiment
    train_name = f"cifar10_gmnet_layers{n_gm_layers}_gaussians{gaussian_value}"
    
    # Create a temporary directory for parameter files if it doesn't exist.
    # This keeps temporary files organized and separate from main parameter files.
    temp_params_dir = os.path.join(os.path.dirname(base_params_path), "temp_experiment_params")
    os.makedirs(temp_params_dir, exist_ok=True)

    # Define the full path for the temporary JSON file
    temp_params_filename = f"temp_gmnet_params_{train_name}.json"
    temp_params_path = os.path.join(temp_params_dir, temp_params_filename)

    # Save the modified parameters to the temporary JSON file
    with open(temp_params_path, 'w') as f:
        json.dump(params, f, indent=4)

    # Construct the command to run train_cifar10.py
    # Feature extraction and loss function are taken from 'train_all.sh' for GMNet.
    command = [
        "python",
        "/media/nas/pgonzalez/gmnet/experiments/cifar10/train_cifar10.py",
        "--train_name", train_name,
        "--dataset", "cifar10",
        "--network", "gmnet",
        "--network_parameters", temp_params_path,
        "--feature_extraction", "nofe",
        "--loss_function", "mrae",
        "--cuda_device", f"cuda:{cuda_device}"
    ]

    print(f"Launching experiment: {train_name} on cuda:{cuda_device}")
    
    # Use subprocess.Popen to run the command in the background (non-blocking).
    # stdout and stderr are captured to check for errors later.
    log_file = open(f"logfiles/log_{train_name}.txt", "w")
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
    
    # Return the process object, the path to the temporary file, and the training name
    # so they can be managed by the main loop.
    return process, temp_params_path, train_name


if __name__ == "__main__":
    # Hardcoded values for n_gm_layers and num_gaussians
    n_gm_layers_list = [9, 3, 6, 12]
    num_gaussians_element_list = [100, 10, 50, 150, 200]
    base_params_path = "/media/nas/pgonzalez/gmnet/experiments/parameters/gmnet_reg001.json"
    cuda_devices_list = [0, 1]  # Example: using CUDA devices 0 and 1
    max_concurrent_runs = 2  # Max 2 concurrent runs as in train_all.sh

    current_device_index = 0
    active_processes_info = [] # Stores (process_object, temp_file_path, train_name) for active runs

    total_experiments = len(n_gm_layers_list) * len(num_gaussians_element_list)
    print(f"Starting {total_experiments} experiments with a maximum of {max_concurrent_runs} concurrent runs.")

    # Iterate through all combinations of n_gm_layers and num_gaussians
    for n_layers in n_gm_layers_list:
        for gaussian_val in num_gaussians_element_list:
            # Check if the maximum number of concurrent runs has been reached
            # If so, wait for an active process to finish before launching a new one.
            while len(active_processes_info) >= max_concurrent_runs:
                # Find if any active process has finished
                finished_a_process = False
                for i, (proc, temp_file, train_name) in enumerate(active_processes_info):
                    if proc.poll() is not None:  # proc.poll() returns None if the process is still running
                        stdout, stderr = proc.communicate() # Get stdout/stderr before removing
                        if proc.returncode != 0:
                            print(f"\nError running {train_name}:")
                            print(stderr)
                        else:
                            print(f"\nSuccessfully finished {train_name}")
                            # Optional: print stdout for successful runs if desired
                            # print(stdout)
                        os.remove(temp_file) # Clean up the temporary JSON file
                        print(f"Removed temporary file: {temp_file}")
                        active_processes_info.pop(i) # Remove finished process from the list
                        finished_a_process = True
                        break # Exit inner loop to re-evaluate 'active_processes_info' length
                
                if not finished_a_process:
                    # If no process finished, wait a short period before checking again
                    time.sleep(5) 

            # Determine the CUDA device for the current experiment by cycling through available devices
            device_id = cuda_devices_list[current_device_index % len(cuda_devices_list)]
            
            # Launch the experiment
            process, temp_file, train_name = run_experiment_process(base_params_path, n_layers, gaussian_val, device_id)
            
            # Add the new process information to the list of active processes
            active_processes_info.append((process, temp_file, train_name))
            current_device_index += 1 # Move to the next device for the next experiment

    # After launching all experiments, wait for any remaining active processes to complete
    print("\nAll experiments launched. Waiting for remaining experiments to finish...")
    for proc, temp_file, train_name in active_processes_info:
        stdout, stderr = proc.communicate() # Wait for process to finish and get outputs
        if proc.returncode != 0:
            print(f"\nError running {train_name}:")
            print(stderr)
        else:
            print(f"\nSuccessfully finished {train_name}")
            # Optional: print stdout for successful runs if desired
            # print(stdout)
        os.remove(temp_file) # Clean up remaining temporary JSON files
        print(f"Removed temporary file: {temp_file}")

    print("\nAll experiments finished.")