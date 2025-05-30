import numpy as np # Used for np.where, np.mean etc.
import matplotlib.pyplot as plt
# It will also need access to the SpatialPooler class if it were to type hint
# or if it did more complex interactions, but for now, it just calls methods on an SP object passed to it.

def run_sp_simulation(sp, input_sdr_list, num_epochs, learn=True):
    """
    Runs a simulation for the Spatial Pooler and collects metrics.
    Args:
        sp (SpatialPooler): The Spatial Pooler instance.
        input_sdr_list (list): A list of input SDRs to train on.
        num_epochs (int): Number of training epochs.
        learn (bool): Whether the SP should learn during processing.
    Returns:
        dict: A dictionary containing lists of metrics per epoch.
    """
    num_inputs = len(input_sdr_list)
    if num_inputs == 0:
        print("No input SDRs provided for simulation.")
        return {}

    # To store SP outputs for each input pattern across epochs for stability check
    # {input_idx: [output_sdr_epoch_0, output_sdr_epoch_1, ...]}
    history_sp_outputs = {i: [] for i in range(num_inputs)}
    
    # Metrics
    avg_output_change_per_epoch = [] # Average Hamming distance of output SDRs for same input vs. previous epoch
    avg_active_column_overlap_per_epoch = [] # Average overlap value of the columns that became active

    for epoch in range(num_epochs):
        current_epoch_active_overlaps = []
        current_epoch_output_changes = []
        
        # Store current epoch's outputs to compare with next one for stability
        current_epoch_outputs_for_stability = [None] * num_inputs

        for i, input_sdr in enumerate(input_sdr_list):
            sp_output_sdr, overlaps = sp.process(input_sdr, learn=learn)
            
            # Metric: Active Column Overlap
            active_column_indices = np.where(sp_output_sdr == 1)[0]
            if len(active_column_indices) > 0:
                current_epoch_active_overlaps.extend(overlaps[active_column_indices])

            # Metric: Output Stability
            if epoch > 0 and len(history_sp_outputs[i]) > 0:
                prev_output_sdr = history_sp_outputs[i][-1] # Output from last epoch for this input
                # Hamming distance = sum of differing bits
                change = np.sum(sp_output_sdr != prev_output_sdr)
                current_epoch_output_changes.append(change)
            
            history_sp_outputs[i].append(sp_output_sdr.copy())
            current_epoch_outputs_for_stability[i] = sp_output_sdr.copy()

        # Aggregate metrics for the epoch
        if current_epoch_active_overlaps:
            avg_active_column_overlap_per_epoch.append(np.mean(current_epoch_active_overlaps))
        else:
            avg_active_column_overlap_per_epoch.append(0) # Avoid error if no columns active

        if current_epoch_output_changes: # Starts from epoch 1
            avg_output_change_per_epoch.append(np.mean(current_epoch_output_changes))
        elif epoch > 0: # If no changes recorded but should have been
             avg_output_change_per_epoch.append(0)


        if (epoch + 1) % (num_epochs // 10 if num_epochs >= 10 else 1) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} completed. "
                  f"Avg Output Change: {avg_output_change_per_epoch[-1] if epoch > 0 else 'N/A'}. "
                  f"Avg Active Overlap: {avg_active_column_overlap_per_epoch[-1]:.2f}")

    # Pad avg_output_change for the first epoch (no change calculable)
    if avg_output_change_per_epoch: # if it has any values
        padded_avg_output_change = [avg_output_change_per_epoch[0]] + avg_output_change_per_epoch
    else: # if it's empty (e.g. num_epochs = 1)
        padded_avg_output_change = [0] * num_epochs


    return {
        "avg_output_change": padded_avg_output_change if len(padded_avg_output_change) == num_epochs else [0]*num_epochs, # ensure correct length
        "avg_active_column_overlap": avg_active_column_overlap_per_epoch
    }

def plot_learning_curves(metrics, num_epochs):
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics["avg_output_change"], label="Avg. Output SDR Change (Hamming)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Hamming Distance")
    plt.title("SP Output Stability")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics["avg_active_column_overlap"], label="Avg. Active Column Overlap Score")
    plt.xlabel("Epoch")
    plt.ylabel("Overlap Score")
    plt.title("SP Active Column Overlap")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
