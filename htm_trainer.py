# htm_trainer.py
import numpy as np
from sdr_utils import get_token_sdr, tokenize # Assuming tokenize is also in sdr_utils


def calculate_sdr_similarity(sdr1, sdr2): # sdr1 is prediction, sdr2 is actual
    print(f"        ++++ ENTERING calculate_sdr_similarity ++++")
    print(f"        DEBUG SIMILARITY Func: sdr1 (pred) input indices = {np.where(sdr1)[0]}, sum={np.sum(sdr1)}, dtype={sdr1.dtype}, shape={sdr1.shape}")
    print(f"        DEBUG SIMILARITY Func: sdr2 (actual) input indices = {np.where(sdr2)[0]}, sum={np.sum(sdr2)}, dtype={sdr2.dtype}, shape={sdr2.shape}")

    if np.sum(sdr1) == 0 and np.sum(sdr2) == 0: 
        print("        DEBUG SIMILARITY Func: Both SDRs empty, returning 1.0")
        return 1.0
    if np.sum(sdr1) == 0 or np.sum(sdr2) == 0: 
        print("        DEBUG SIMILARITY Func: One SDR empty, returning 0.0")
        return 0.0
    
    sdr1_bool = sdr1.astype(bool)
    sdr2_bool = sdr2.astype(bool)
    
    # VVVV CORRECTED OVERLAP CALCULATION VVVV
    overlap = np.sum(sdr1_bool & sdr2_bool) 
    # VVVV END OF CORRECTION VVVV
    
    sum_sdr2_active_bits = np.sum(sdr2_bool)

    print(f"        DEBUG SIMILARITY Func: sdr1_bool indices (after astype) = {np.where(sdr1_bool)[0]}")
    print(f"        DEBUG SIMILARITY Func: sdr2_bool indices (after astype) = {np.where(sdr2_bool)[0]}")
    print(f"        DEBUG SIMILARITY Func: Calculated overlap (using np.sum(sdr1_bool & sdr2_bool)) = {overlap}") 
    print(f"        DEBUG SIMILARITY Func: Sum of active in actual (sum_sdr2_active_bits) = {sum_sdr2_active_bits}")
    
    if sum_sdr2_active_bits == 0: 
        print("        DEBUG SIMILARITY Func: Sum of active in actual is 0, returning 0.0 to avoid division error")
        return 0.0
        
    calculated_sim = float(overlap) / float(sum_sdr2_active_bits) 
    print(f"        DEBUG SIMILARITY Func: Raw calculated similarity = {calculated_sim:.2f}")
    print(f"        ---- EXITING calculate_sdr_similarity ----")
    return calculated_sim


def run_htm_training_loop(sp_model, tm_model, sequences, sdr_params_dict,
                          num_epochs, sp_output_to_token_map_for_decode=None, 
                          min_prediction_overlap_threshold=0.8,
                          sp_learning_epochs=5):
    """
    Runs an automated training loop for the SP and TM.
    Args:
        sp_model: The SpatialPooler instance.
        tm_model: The TemporalMemory instance.
        sequences (list of lists of str): Training data.
        sdr_params_dict (dict): Parameters for get_token_sdr.
        num_epochs (int): Number of times to iterate through the dataset.
        sp_output_to_token_map_for_decode (dict, optional): For printing.
        min_prediction_overlap_threshold (float): Threshold for correct TM prediction.
        sp_learning_epochs (int): Number of initial epochs during which the SP learns.
    """
    
    print(f"\n--- Starting HTM Training Loop for {num_epochs} epochs ---")
    print(f"    Spatial Pooler will learn for the first {sp_learning_epochs} epoch(s).")

    for epoch in range(num_epochs):
        total_predictions = 0
        correct_predictions = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        sp_should_learn_this_epoch = (epoch < sp_learning_epochs)
        if epoch == sp_learning_epochs:
            print(f"    SP learning is now OFF for subsequent epochs.")

        for seq_idx, token_sequence in enumerate(sequences):
            if not token_sequence:
                continue

            tm_model.prev_active_cells.fill(False)
            tm_model.prev_winner_cells.fill(False)
            tm_model.predictive_cells.fill(False)

            actual_sp_outputs_in_sequence = []
            for token in token_sequence:
                token_sdr = get_token_sdr(token, sdr_params_dict['sdr_size'], sdr_params_dict['active_bits'])
                sp_output, _ = sp_model.process(token_sdr, learn=sp_should_learn_this_epoch) 
                actual_sp_outputs_in_sequence.append(sp_output)

            for t in range(len(actual_sp_outputs_in_sequence)):
                current_sp_sdr = actual_sp_outputs_in_sequence[t]
                tm_all_predictive_cells_sdr = tm_model.process(current_sp_sdr, learn=True)

                if t < len(actual_sp_outputs_in_sequence) - 1:
                    actual_next_sp_column_sdr = actual_sp_outputs_in_sequence[t+1]
                    total_predictions += 1

                    #CONVERT TM CELL PREDICTION TO COLUMN PREDICTION
                    predicted_next_columns_sdr = np.zeros(tm_model.num_columns, dtype=int)
                    if np.any(tm_all_predictive_cells_sdr > 0): # Check for any positive strength
                        # CHANGE: Get indices where strength is greater than 0
                        predicted_cell_indices = np.where(tm_all_predictive_cells_sdr > 0)[0]
                        for abs_cell_idx in predicted_cell_indices:
                            col_idx = abs_cell_idx // tm_model.cells_per_column
                            predicted_next_columns_sdr[col_idx] = 1
                    
                    # CHANGE: The debug print for TM_Pred_CELL_SDR should reflect it's showing indices with strength > 0
                    print(f"      DEBUG ACCURACY: TM_Pred_CELL_SDR (indices with strength > 0): {np.where(tm_all_predictive_cells_sdr > 0)[0]}") 
                    print(f"      DEBUG ACCURACY: TM_Pred_COLUMN_SDR (for similarity func): {np.where(predicted_next_columns_sdr)[0]}")
                    print(f"      DEBUG ACCURACY: Actual_Next_COLUMN_SDR (for similarity func): {np.where(actual_next_sp_column_sdr)[0]}")
                    
                    similarity = calculate_sdr_similarity(predicted_next_columns_sdr, actual_next_sp_column_sdr)

                    if similarity >= min_prediction_overlap_threshold:
                        correct_predictions += 1
                        print(f"      ---> CORRECT PREDICTION! Sim: {similarity:.2f}")
                    else:
                        print(f"      ---> WRONG PREDICTION. Sim: {similarity:.2f}. Pred: {np.where(predicted_next_columns_sdr)[0]}, Actual: {np.where(actual_next_sp_column_sdr)[0]}")                        
                    
                    # Conditional print for detailed sequence step (for first sequence of sampled epochs)
                    if sp_output_to_token_map_for_decode and epoch % (num_epochs // 5 if num_epochs >=5 else 1) == 0 and seq_idx == 0:
                        current_token_str = token_sequence[t]
                        actual_next_token_str = token_sequence[t+1]
                        # This decoding is for human readability and might be imperfect if map is not exhaustive
                        # pred_token_str = sp_output_to_token_map_for_decode.get(tuple(predicted_next_columns_sdr.tolist()), "(decoding_unknown)")
                        
                        print(f"  Seq {seq_idx}, t={t}: Input '{current_token_str}' (SP: {np.where(current_sp_sdr)[0]})")
                        print(f"    TM Predicted Next SP SDR: {np.where(predicted_next_columns_sdr)[0]}")
                        # print(f"    (Decoded TM Pred: '{pred_token_str}')") # Optional
                        print(f"    Actual Next SP SDR    : {np.where(actual_next_sp_column_sdr)[0]} ('{actual_next_token_str}') -> Correct: {similarity >= min_prediction_overlap_threshold} (Sim: {similarity:.2f})")

            if len(sequences) > 1 and seq_idx % (len(sequences) // 10 if len(sequences) >= 10 else 1) == 0 and seq_idx > 0 : # Avoid print if only 1 sequence
                 print(f"  Processed {seq_idx + 1}/{len(sequences)} sequences...")

        epoch_accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
        print(f"Epoch {epoch + 1} completed. Prediction Accuracy: {epoch_accuracy:.4f} ({correct_predictions}/{total_predictions})")

        if epoch_accuracy == 1.0 and total_predictions > 0 and epoch >= sp_learning_epochs : 
            print("Perfect accuracy achieved after SP stabilization. Stopping training.")
            break

    print("--- HTM Training Loop Finished ---")