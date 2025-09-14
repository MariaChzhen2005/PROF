import os
import numpy as np
import torch
from tqdm import tqdm
import uuid # For unique simulation IDs if needed, though less critical now

import gym
import eplus_env # Ensure this is importable if not automatically handled by gym.make

# --- Configuration ---
# OUTPUT_DATA_DIR: Directory to save the generated .pt and .npz files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA_DIR = os.path.join(SCRIPT_DIR, "controller_inputs_building")

# Feasibility constraints for operative temperature
T_OP_MIN_FEASIBLE = 21.9  # degrees Celsius
T_OP_MAX_FEASIBLE = 25.5  # degrees Celsius

# --- Simulation Parameters (User to define) ---
# Define ranges for ZOME THERMOSTAT SETPOINTS (actions for the environment)
# These are examples; please adjust based on your requirements.
ZONE_HEATING_SETPOINT_VALS = np.linspace(20, 26, 4)  # Actions: Min 20 C. e.g., 20, 22, 24, 26 degrees C
ZONE_COOLING_SETPOINT_VALS = np.linspace(22, 28, 4)  # Actions: e.g., 22, 24, 26, 28 degrees C (must be > heating)

N_STEPS_PER_TRIAL = 12 # Number of 5-minute env steps to run for each setpoint combination (12 steps = 1 hour)

# Indices for extracting relevant data from the observation vector `obs`
# Based on the order in the variables.cfg we examined.
# THIS ORDER IS CRUCIAL AND MUST MATCH THE ENV'S ACTUAL OBSERVATION SPACE.
# PLEASE VERIFY THIS WITH YOUR `Eplus-IW-test-v0` environment's specific configuration.
OBS_IDX_T_OA = 0           # Site Outdoor Air Drybulb Temperature
OBS_IDX_RH_OA = 1          # Site Outdoor Air Relative Humidity
OBS_IDX_WIND_SPEED = 2     # Site Wind Speed
OBS_IDX_WIND_DIR = 3       # Site Wind Direction
OBS_IDX_SOLAR_DIFFUSE = 4  # Site Diffuse Solar Radiation Rate per Area
OBS_IDX_SOLAR_DIRECT = 5   # Site Direct Solar Radiation Rate per Area
# OBS_IDX_ZONE_HTG_SP_OUT = 6 # Zone Thermostat Heating Setpoint Temperature (output from E+)
# OBS_IDX_ZONE_CLG_SP_OUT = 7 # Zone Thermostat Cooling Setpoint Temperature (output from E+)
OBS_IDX_ZONE_AIR_TEMP = 8  # SPACE1-1 Zone Air Temperature (our proxy for T_op)
# OBS_IDX_MRT = 9            # Mean Radiant Temperature
# OBS_IDX_ZONE_RH = 10       # Zone Air Relative Humidity
# OBS_IDX_CLOTHING = 11      # Clothing Value
# OBS_IDX_PPD = 12           # Fanger PPD
# OBS_IDX_OCC_COUNT = 13     # Occupant Count
OBS_IDX_HVAC_POWER = 14    # Facility Total HVAC Electric Demand Power


def generate_data(env, heating_setpoints, cooling_setpoints, num_steps_per_trial):
    """
    Generates building simulation data using the Gym environment by varying
    zone thermostat setpoints.
    """
    print("Starting data generation using Gym environment...")
    all_data_points = []
    feasibility_labels = []

    total_combinations = 0
    # Pre-calculate valid combinations to give accurate total for tqdm
    valid_combinations = []
    for h_sp_test in heating_setpoints:
        for c_sp_test in cooling_setpoints:
            if h_sp_test < c_sp_test: # Valid if heating SP < cooling SP
                valid_combinations.append((h_sp_test, c_sp_test))
    
    total_combinations = len(valid_combinations)
    print(f"Total valid setpoint combinations to simulate: {total_combinations}")

    for h_sp, c_sp in tqdm(valid_combinations, desc="Simulating Setpoint Combinations"):
        action = [h_sp, c_sp] # Action: [heating_setpoint, cooling_setpoint]
        
        try:
            _ = env.reset()
            final_obs_for_trial = None
            for step_num in range(num_steps_per_trial):
                obs, reward, done, info = env.step(action)
                final_obs_for_trial = obs
                if done:
                    break
            
            if final_obs_for_trial is not None:
                T_oa_obs = final_obs_for_trial[OBS_IDX_T_OA]
                RH_oa_obs = final_obs_for_trial[OBS_IDX_RH_OA]
                Wind_Speed_obs = final_obs_for_trial[OBS_IDX_WIND_SPEED]
                Wind_Dir_obs = final_obs_for_trial[OBS_IDX_WIND_DIR]
                Solar_direct_obs = final_obs_for_trial[OBS_IDX_SOLAR_DIRECT]
                Solar_diffuse_obs = final_obs_for_trial[OBS_IDX_SOLAR_DIFFUSE]
                
                Zone_Air_Temp_obs = final_obs_for_trial[OBS_IDX_ZONE_AIR_TEMP]
                HVAC_Power_obs = final_obs_for_trial[OBS_IDX_HVAC_POWER]

                # Construct the state vector:
                # [Actions (2: H_SP, C_SP), Observed Env (6), Observed Outputs (2)] -> Total 10 features
                state_vector = [
                    h_sp, c_sp,
                    T_oa_obs, RH_oa_obs, Wind_Speed_obs, Wind_Dir_obs, 
                    Solar_direct_obs, Solar_diffuse_obs, # Env Conditions
                    Zone_Air_Temp_obs, HVAC_Power_obs  # System Outputs
                ]
                all_data_points.append(state_vector)

                is_feasible = (T_OP_MIN_FEASIBLE <= Zone_Air_Temp_obs <= T_OP_MAX_FEASIBLE)
                feasibility_labels.append(is_feasible)
            else:
                print(f"Warning: No final observation obtained for H={h_sp:.1f}, C={c_sp:.1f}. Skipping.")
        except Exception as e:
            print(f"Error for H={h_sp:.1f}, C={c_sp:.1f}: {e}. Skipping.")

    print(f"Finished simulations. Generated {len(all_data_points)} data points.")
    return np.array(all_data_points), np.array(feasibility_labels).astype(bool)

def save_processed_data(all_points_np, feasibility_labels_np, output_dir):
    """
    Processes the generated data and saves it in PyTorch tensors and .npz format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if all_points_np.shape[0] == 0:
        print("No data points generated. Skipping saving.")
        return

    X_all_tensor = torch.FloatTensor(all_points_np)
    # Ensure feasibility_labels_np is boolean before converting for safety, though astype(bool) should handle it
    feasible_mask_tensor = torch.BoolTensor(feasibility_labels_np.astype(bool))

    torch.save(X_all_tensor, os.path.join(output_dir, "X_all_building.pt"))
    torch.save(feasible_mask_tensor, os.path.join(output_dir, "feasible_mask_building.pt"))
    print(f"Saved X_all_building.pt (shape: {X_all_tensor.shape})")
    print(f"Saved feasible_mask_building.pt (shape: {feasible_mask_tensor.shape})")

    if X_all_tensor.numel() > 0: # Check if tensor is not empty
        X_feasible_tensor = X_all_tensor[feasible_mask_tensor]
        X_infeasible_tensor = X_all_tensor[~feasible_mask_tensor]

        torch.save(X_feasible_tensor, os.path.join(output_dir, "X_feasible_building.pt"))
        torch.save(X_infeasible_tensor, os.path.join(output_dir, "X_infeasible_building.pt"))
        print(f"Saved X_feasible_building.pt (shape: {X_feasible_tensor.shape})")
        print(f"Saved X_infeasible_building.pt (shape: {X_infeasible_tensor.shape})")

        # Calculate and save normalization parameters (mean and std) from X_all
        mean_params = torch.mean(X_all_tensor, dim=0).numpy()
        std_params = torch.std(X_all_tensor, dim=0).numpy()
        std_params[std_params < 1e-6] = 1e-6 
        
        np.savez_compressed(
            os.path.join(output_dir, "normalization_params_building.npz"),
            mean=mean_params,
            std=std_params
        )
        print(f"Saved normalization_params_building.npz (mean shape: {mean_params.shape}, std shape: {std_params.shape})")
    else:
        print("X_all_tensor is empty. Skipping feasible/infeasible split and normalization params.")


if __name__ == "__main__":
    print("--- Building Data Generation using Gym Environment ---")
    print("IMPORTANT: This script assumes 'Eplus-IW-test-v0' is correctly installed and configured.")
    print("It controls ZONE THERMOSTAT SETPOINTS, not direct supply air/water temperatures.")
    print("Environmental conditions are OBSERVED from the environment (EPW-driven).")
    print("PLEASE VERIFY:")
    print("  1. Ranges for ZONE_HEATING_SETPOINT_VALS and ZONE_COOLING_SETPOINT_VALS.")
    print("  2. N_STEPS_PER_TRIAL (how many 5-min steps per setpoint trial).")
    print("  3. Observation indices (OBS_IDX_*) match your environment's observation space.")
    print("  4. Your EnergyPlus/BCVTB/Java setup for eplus_env is functional.")
    print("  5. Run 'find_eplus_env.py' to confirm the active E+ model configuration.")
    print("---------------------------------------------------------")

    try:
        env = gym.make('Eplus-IW-test-v0')
        print(f"Successfully created Gym environment: Eplus-IW-test-v0")
        print(f"  Action Space: {env.action_space}")
        print(f"  Observation Space: {env.observation_space}")
        # You might want to check env.action_space.shape and env.observation_space.shape
        # to ensure they align with expectations (e.g., action dim 2, obs dim >= 15)
        
        # Check if observation space seems compatible with OBS_IDX constants
        if isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.shape[0] < max(OBS_IDX_T_OA, OBS_IDX_RH_OA, OBS_IDX_SOLAR_DIRECT, OBS_IDX_SOLAR_DIFFUSE, OBS_IDX_ZONE_AIR_TEMP, OBS_IDX_HVAC_POWER) + 1:
            print("CRITICAL WARNING: Observation space dimension is smaller than expected based on OBS_IDX constants!")
            print("  Please verify OBS_IDX constants against the actual observation vector structure.")
            # exit()

        all_points, feasibility_mask = generate_data(
            env, 
            ZONE_HEATING_SETPOINT_VALS, 
            ZONE_COOLING_SETPOINT_VALS, 
            N_STEPS_PER_TRIAL
        )
        
        if all_points.size > 0:
            save_processed_data(all_points, feasibility_mask, OUTPUT_DATA_DIR)
            print("Building data generation and saving complete.")
        else:
            print("No data was generated. Check simulation settings, environment, and setpoint ranges.")
        
        env.close()

    except gym.error.NameNotFound:
        print("ERROR: Gym environment 'Eplus-IW-test-v0' not found.")
        print("  Please ensure 'eplus_env' is correctly installed and the environment is registered.")
    except Exception as e:
        print(f"An error occurred during the data generation process: {e}")
        import traceback
        traceback.print_exc() 