# chat.py
import numpy as np
import os
import sys
import datetime # For logging timestamps

# Assuming your custom modules are in the python path or same directory
from sdr_utils import tokenize, get_token_sdr, _token_sdr_cache, clear_sdr_cache
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory # Ensure this uses conditional printing or has prints commented out for clean chat
from persistence import load_state

def run_chat():
    # --- Configurations (should match the training configurations) ---
    sdr_config = {
        'sdr_size': 256,
        'active_bits': 10
    }
    sp_config = {
        'input_size': sdr_config['sdr_size'],
        'num_columns': 128,
        'num_active_columns': 5, # Crucial for decoding
        'permanence_threshold': 0.5,
        'initial_permanence_std_dev': 0.1,
        'permanence_increment': 0.1,
        'permanence_decrement': 0.02,
        'learning_rate': 0.1,
        'global_inhibition': True,
        'seed': 42
    }
    tm_config = {
        'num_columns': sp_config['num_columns'],
        'cells_per_column': 4,
        'activation_threshold': 3,
        'initial_permanence': 0.21,
        'connected_permanence': 0.50,
        'learning_threshold': 2, # From your successful fast-learning run
        'permanence_increment': 0.10,
        'permanence_decrement': 0.05,
        'max_synapses_per_segment': 32,
        'max_segments_per_cell': 128,
        'seed': 123
        # 'verbose': False # Set this if you implement conditional verbosity in TemporalMemory
    }
    main_state_file = 'htm_main_model_state.pkl'
    chat_log_filename = f"chat_interaction_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print(f"Chat interactions will be logged to: {chat_log_filename}")

    # --- Load Model ---
    if not os.path.exists(main_state_file):
        print(f"Error: Model state file '{main_state_file}' not found. Please train the model first.")
        return

    print("Loading HTM model state...")
    sp_main, tm_main, loaded_token_to_input_sdr_map = load_state(
        main_state_file, sp_config, tm_config, sdr_config
    )
    
    if loaded_token_to_input_sdr_map:
        _token_sdr_cache.clear() 
        _token_sdr_cache.update(loaded_token_to_input_sdr_map)
        print(f"Loaded {len(_token_sdr_cache)} token-to-input-SDR mappings into sdr_utils cache.")
    else:
        print("Warning: No token-to-input-SDR map loaded from state.")


    # --- Build SP Output to Token Map ---
    sp_output_to_token_map = {}
    if sp_main and _token_sdr_cache:
        print("Reconstructing SP output to token map...")
        for token, input_sdr_arr in _token_sdr_cache.items():
            current_input_sdr = np.array(input_sdr_arr) if not isinstance(input_sdr_arr, np.ndarray) else input_sdr_arr
            sp_output_sdr, _ = sp_main.process(current_input_sdr, learn=False)
            sp_output_to_token_map[tuple(sp_output_sdr.tolist())] = token
        print(f"Reconstructed map for {len(sp_output_to_token_map)} SP outputs.")
    
    if not sp_output_to_token_map:
        print("Could not build SP output to token map. Decoding will likely fail.")

    print("\nHTM Chat Ready. Type 'quit' to exit.")
    
    try:
        with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
            chat_f.write(f"--- Chat Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            chat_f.write(f"Model: {main_state_file}\n")
            chat_f.write(f"SP Config: {sp_config}\n")
            chat_f.write(f"TM Config: {tm_config}\n")
            chat_f.write("------\n")

        while True:
            user_input_text = input("You: ")
            with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
                chat_f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} You: {user_input_text}\n")

            if user_input_text.lower() == 'quit':
                break
            if not user_input_text.strip():
                print("Bot: Please say something.")
                with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
                    chat_f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Bot: Please say something.\n")
                continue

            tokens = tokenize(user_input_text)
            if not tokens:
                print("Bot: I couldn't understand the words in that. Try again?")
                with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
                    chat_f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Bot: I couldn't understand the words in that. Try again?\n")
                continue
            
            # Reset TM for new input context
            tm_main.prev_active_cells.fill(False)
            tm_main.prev_winner_cells.fill(False)
            tm_main.predictive_cells.fill(0) 

            # Prime TM with user's input
            current_sp_sdr_for_tm_processing = None 
            
            for token_idx, token in enumerate(tokens):
                input_sdr = get_token_sdr(token, sdr_config['sdr_size'], sdr_config['active_bits'])
                current_sp_sdr_for_tm_processing, _ = sp_main.process(input_sdr, learn=False)
                _ = tm_main.process(current_sp_sdr_for_tm_processing, learn=False) 

            # Generate response
            generated_response_tokens = []
            max_gen_tokens = 15 

            for gen_idx in range(max_gen_tokens):
                if not np.any(tm_main.predictive_cells > 0):
                    if gen_idx == 0: generated_response_tokens.append("[No initial prediction from TM]")
                    # else: generated_response_tokens.append("[Prediction faded]") # Already handled by loop break
                    break 

                column_prediction_strengths = np.zeros(tm_main.num_columns, dtype=float)
                predicted_cell_indices = np.where(tm_main.predictive_cells > 0)[0]

                if not predicted_cell_indices.size > 0:
                    generated_response_tokens.append("[No cells had predictive strength]")
                    break

                for abs_cell_idx in predicted_cell_indices:
                    col_idx = abs_cell_idx // tm_main.cells_per_column
                    column_prediction_strengths[col_idx] += tm_main.predictive_cells[abs_cell_idx]

                predicted_next_columns_sdr = np.zeros(tm_main.num_columns, dtype=int)
                
                if not np.any(column_prediction_strengths > 0):
                    generated_response_tokens.append("[Zero column strength after aggregation]")
                    break

                num_target_active_columns = sp_config.get('num_active_columns', 5)
                columns_with_any_strength_indices = np.where(column_prediction_strengths > 0)[0]

                # --- Option 2: Stop if prediction is too sparse ---
                if columns_with_any_strength_indices.size < num_target_active_columns:
                    generated_response_tokens.append(f"[Pred. too sparse: {columns_with_any_strength_indices.size} distinct col(s) with strength < {num_target_active_columns} target]")
                    break 
                
                # If we reach here, we have enough columns with strength to pick top N
                noisy_column_strengths = column_prediction_strengths + np.random.uniform(-1e-9, 1e-9, size=column_prediction_strengths.shape)
                top_n_column_indices = np.argsort(noisy_column_strengths)[-num_target_active_columns:]
                predicted_next_columns_sdr[top_n_column_indices] = 1
                # --- End of forming predicted_next_columns_sdr ---

                sdr_tuple_to_lookup = tuple(predicted_next_columns_sdr.tolist())
                predicted_indices = np.where(predicted_next_columns_sdr > 0)[0]
                
                # [CHAT DEBUG] print - keeping this active as requested
                print(f"\n[CHAT DEBUG] Attempting to decode SP SDR: {predicted_indices} (SDR tuple: {sdr_tuple_to_lookup})")
                
                predicted_token = sp_output_to_token_map.get(sdr_tuple_to_lookup)

                if predicted_token:
                    generated_response_tokens.append(predicted_token)
                    if predicted_token == '.' or len(generated_response_tokens) >= max_gen_tokens:
                        break 
                    
                    next_input_sdr = get_token_sdr(predicted_token, sdr_config['sdr_size'], sdr_config['active_bits'])
                    current_sp_sdr_for_tm_processing, _ = sp_main.process(next_input_sdr, learn=False)
                    _ = tm_main.process(current_sp_sdr_for_tm_processing, learn=False)
                else:
                    generated_response_tokens.append(f"[SDR Not In Map: {predicted_indices}]")
                    break 
            
            bot_response_str = " ".join(generated_response_tokens)
            print(f"Bot: {bot_response_str}", flush=True)
            with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
                chat_f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Bot: {bot_response_str}\n")

    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print(f"An error occurred in chat loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with open(chat_log_filename, 'a', encoding='utf-8') as chat_f:
            chat_f.write(f"--- Chat Session Ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
        print(f"Chat session logged to {chat_log_filename}")


if __name__ == '__main__':
    # If you want to clear the SDR cache at the start of a chat session (optional)
    # clear_sdr_cache() 
    run_chat()