import numpy as np
import matplotlib.pyplot as plt # Still useful if you save plots to files
import os
import time
import sys
import datetime


# Imports from your custom modules
import sdr_utils 
from sdr_utils import tokenize, get_token_sdr, clear_sdr_cache # Removed create_sdr as it's internal to get_token_sdr
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from persistence import save_state, load_state
from htm_trainer import run_htm_training_loop # calculate_sdr_similarity is used by run_htm_training_loop

# --- Tee Class for Logging (copied from notebook) ---
class Tee(object):
    def __init__(self, filename_prefix="htm_output_log", mode='a'):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{filename_prefix}_{timestamp}.txt"
        self.file = open(self.log_filename, mode, encoding='utf-8')
        print(f"--- Logging output to: {self.log_filename} ---", flush=True) # Added flush
        sys.stdout = self
        # sys.stderr = self # Optional

    def __del__(self):
        self.close()

    def close(self):
        original_stdout_ref = self.stdout # Store before potentially overwriting sys.stdout
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            # Ensure we print to the actual original console, not potentially self.file again
            if original_stdout_ref:
                print(f"--- Closed log file: {self.log_filename} ---", file=original_stdout_ref, flush=True)
            self.file.close()
            self.file = None

    def write(self, data):
        if self.file and not self.file.closed:
            self.file.write(data)
        if self.stdout and hasattr(self.stdout, 'write'):
            try:
                self.stdout.write(data)
            except Exception as e: # Fallback if original stdout is somehow problematic
                pass


    def flush(self):
        if self.file and not self.file.closed:
            self.file.flush()
        if self.stdout and hasattr(self.stdout, 'flush'):
            self.stdout.flush()

    def fileno(self):
        return self.stdout.fileno() if self.stdout else -1

def main():
    # Activate logging
    # Keep logger as a local variable in main, it will be closed when main exits or __del__ is called
    logger = Tee()

    print("Modules imported and logger activated.")

    # --- Global Parameters ---
    sdr_config = {
        'sdr_size': 256,
        'active_bits': 10
    }
    sp_config = {
        'input_size': sdr_config['sdr_size'],
        'num_columns': 128,
        'num_active_columns': 5,
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
        'learning_threshold': 2, # Using the successful setting
        'permanence_increment': 0.10,
        'permanence_decrement': 0.05,
        'max_synapses_per_segment': 32,
        'max_segments_per_cell': 128,
        'seed': 123
    }
    main_state_file = 'htm_main_model_state.pkl'

    print("\nParameters defined:")
    print(f"  SDR size: {sdr_config['sdr_size']}, Active bits: {sdr_config['active_bits']}")
    print(f"  SP columns: {sp_config['num_columns']}, Active SP columns: {sp_config['num_active_columns']}")
    print(f"  TM columns: {tm_config['num_columns']}, Cells per column: {tm_config['cells_per_column']}")
    print(f"  TM learning_threshold: {tm_config['learning_threshold']}")


    # --- Delete old state file for a clean run (IMPORTANT FOR TESTING) ---
    # if os.path.exists(main_state_file):
    #     print(f"\n--- Deleting existing state file: {main_state_file} for a fresh start ---")
    #     os.remove(main_state_file)
    # --- End of Deletion ---


    # --- Load or Initialize Main SP and TM ---
    print(f"\nAttempting to load/initialize state from: {main_state_file}")
    sp_main, tm_main, loaded_token_sdr_map = load_state(
        main_state_file,
        sp_config,
        tm_config,
        sdr_config
    )
    # The load_state function should print "No state file ... found. Initializing new state."
    # and the TM __init__ prints should confirm fresh initialization.

    if loaded_token_sdr_map:
        print(f"Loaded {len(loaded_token_sdr_map)} token SDRs from state file into sdr_utils cache.")
        clear_sdr_cache()
        # Update the module-level cache in sdr_utils
        # Need to import sdr_utils itself to access its _token_sdr_cache
        # This import is already at the top, so we can directly use sdr_utils.
        sdr_utils._token_sdr_cache.update(loaded_token_sdr_map)
    else:
        print("No existing token SDR map loaded, or map was empty. sdr_utils cache will populate as new tokens are seen.")

    print("\nMain SP and TM instances are ready.")

    # --- Automated HTM Training ---
    print("\n--- Setting up for Automated HTM Training on Text Data ---")

    # 1. Load and Preprocess Your Text Data
    # Example: Replace this with your actual text loading and processing
    raw_text = """
    The quick brown fox jumps over the lazy dog.
    A lazy dog may not notice a quick brown fox.
    The fox is quick and the dog is lazy.
    """
    # Simple sentence splitting and tokenization
    # More robust sentence splitting might be needed for complex texts (e.g., using NLTK)
    sentences = [s.strip() for s in raw_text.strip().split('.') if s.strip()]
    training_sequences = [tokenize(sentence) for sentence in sentences if tokenize(sentence)] 
    # Ensure no empty sequences if sentences were just punctuation.

    # Optional: Repeat the text data to create more training instances if your corpus is small
    # original_training_sequences = list(training_sequences) # Make a copy
    # num_text_repetitions = 100
    # training_sequences = [seq for _ in range(num_text_repetitions) for seq in original_training_sequences]


    print(f"Prepared {len(training_sequences)} sequences for training.")
    if training_sequences:
        print(f"Example sequence: {training_sequences[0]}")

    num_training_epochs = 200 # Adjust based on corpus size and desired training intensity
    initial_sp_learning_epochs = 5 # SP learns vocabulary in first few passes

    # Pre-populate sp_output_to_token_map for debug printing 
    automated_sp_output_to_token_map = {}
    print("Pre-populating sp_output_to_token_map for debug printing...")
    all_tokens_in_training = list(set(token for seq in training_sequences for token in seq))
    for token in all_tokens_in_training:
        sdr = get_token_sdr(token, sdr_config['sdr_size'], sdr_config['active_bits'])
        sp_output_for_token, _ = sp_main.process(sdr, learn=False)
        automated_sp_output_to_token_map[tuple(sp_output_for_token.tolist())] = token
    print(f"Built a map for {len(automated_sp_output_to_token_map)} SP outputs to tokens.")

    # Call the training loop
    if training_sequences:
        run_htm_training_loop(
            sp_model=sp_main,
            tm_model=tm_main,
            sequences=training_sequences,
            sdr_params_dict=sdr_config,
            num_epochs=num_training_epochs,
            sp_output_to_token_map_for_decode=automated_sp_output_to_token_map,
            min_prediction_overlap_threshold=0.8, 
            sp_learning_epochs=initial_sp_learning_epochs
        )
    else:
        print("No training sequences to process.")


    # After training, save the state
    print("\nSaving model state after automated training...")
    # Ensure the sdr_utils global cache is what gets saved in token_sdr_map
    save_state(sp_main, tm_main, sdr_utils._token_sdr_cache, main_state_file)
    print("Automated training and state saving complete.")

    # Explicitly close logger if you want to see "Log file closed" message before script ends
    # logger.close() # Tee's __del__ will also try to close it.

if __name__ == '__main__':
    main()
    # If logger was global, ensure it's closed after main finishes or on script exit
    # This can be tricky if main() raises an exception. The __del__ in Tee is a fallback.
    if 'logger' in globals() and isinstance(logger, Tee): # Check if logger was defined (it's local to main now)
         pass # logger is local to main, __del__ will handle it.
             # If it were global and assigned outside main, you might do logger.close() here.


