import pickle
import os
import numpy as np # For np.array when loading TM states
from spatial_pooler import SpatialPooler # Because load_state instantiates it
from temporal_memory import TemporalMemory # Because load_state instantiates it
# from sdr_utils import clear_sdr_cache (If you decide to manage the cache from here)

# --- Persistence functions (keep here or move to persistence.py) ---
# If you move them, they'll need to take SP, TM objects as args and import pickle
def save_state(spatial_pooler_obj, temporal_memory_obj, token_map, file_name):
    # ... (implementation using spatial_pooler_obj.permanences etc.)
    # Will need careful thought on how to save TM's self.segments
    tm_segments_serializable = []
    if hasattr(temporal_memory_obj, 'segments'): # Check if TM is properly initialized
        for cell_segments in temporal_memory_obj.segments:
            cell_data = []
            for seg in cell_segments:
                # Convert synapse dict keys (numpy int if from np.where) to standard int for pickle/json
                serializable_synapses = {int(k): float(v) for k, v in seg['synapses'].items()}
                cell_data.append({'synapses': serializable_synapses, 'is_sequence_segment': seg['is_sequence_segment']})
            tm_segments_serializable.append(cell_data)

    with open(file_name, 'wb') as f:
        pickle.dump({
            'spatial_pooler_permanences': spatial_pooler_obj.permanences if spatial_pooler_obj else None,
            'sp_column_activations_count': spatial_pooler_obj.column_activations_count if spatial_pooler_obj else None,
            # TM state saving
            'tm_segments': tm_segments_serializable if temporal_memory_obj else None,
            'tm_prev_active_cells': temporal_memory_obj.prev_active_cells.tolist() if temporal_memory_obj and hasattr(temporal_memory_obj, 'prev_active_cells') else None, # Convert to list
            'tm_prev_winner_cells': temporal_memory_obj.prev_winner_cells.tolist() if temporal_memory_obj and hasattr(temporal_memory_obj, 'prev_winner_cells') else None, # Convert to list
            'token_sdr_map': token_map # This is already _token_sdr_cache, maybe save/load that directly or pass it
        }, f)
    print(f"State saved to {file_name}")
    # show_file_info(file_name) # if you keep it here

def load_state(file_name, sp_params, tm_params, sdr_params): # Pass dicts of params
    # sp_params = {'input_size': ..., 'num_columns': ..., ...}
    # tm_params = {'num_columns': ..., 'cells_per_column': ..., ...}
    # sdr_params = {'sdr_size': ..., 'active_bits': ...}

    # Initialize with default structures first
    sp = SpatialPooler(**sp_params) # Unpack dict into arguments
    tm = TemporalMemory(**tm_params)
    
    # For token_sdr_map, we are using the module-level _token_sdr_cache in sdr_utils.
    # So, we might not need to explicitly load/save it here if sdr_utils manages its persistence
    # or if we pass the cache around. For now, let's assume sdr_utils cache is transient per session
    # unless we add save/load to sdr_utils itself.
    # Let's re-introduce a token_map at this level for explicit save/load.
    loaded_token_map = {}


    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        # show_file_info(file_name)
        
        if data.get('spatial_pooler_permanences') is not None:
            sp.permanences = data['spatial_pooler_permanences']
        if data.get('sp_column_activations_count') is not None:
            sp.column_activations_count = data['sp_column_activations_count']

        # Load TM state
        loaded_segments_serializable = data.get('tm_segments')
        if loaded_segments_serializable:
            tm.segments = [[] for _ in range(tm.num_cells)] # Reinitialize
            for cell_idx, cell_data_serializable in enumerate(loaded_segments_serializable):
                for seg_data_serializable in cell_data_serializable:
                    # Keys in synapses are already int from saving logic
                    tm.segments[cell_idx].append({
                        'synapses': seg_data_serializable['synapses'],
                        'is_sequence_segment': seg_data_serializable['is_sequence_segment']
                    })
        if data.get('tm_prev_active_cells') is not None:
            tm.prev_active_cells = np.array(data['tm_prev_active_cells'], dtype=bool)
        if data.get('tm_prev_winner_cells') is not None:
            tm.prev_winner_cells = np.array(data['tm_prev_winner_cells'], dtype=bool)
        
        loaded_token_map = data.get('token_sdr_map', {})
        # If using sdr_utils._token_sdr_cache, you might want to update it:
        # sdr_utils._token_sdr_cache.update(loaded_token_map)
        print(f"Loaded state from {file_name}")
    else:
        print(f"No state file {file_name} found. Initializing new state.")
    
    # For get_token_sdr, we still rely on the cache in sdr_utils.
    # To make loaded_token_map effective for get_token_sdr, we would need to
    # either pass loaded_token_map to get_token_sdr or update sdr_utils._token_sdr_cache.
    # Let's assume for now that sdr_utils._token_sdr_cache is what get_token_sdr uses,
    # and this loaded_token_map is for other purposes or needs to sync with the cache.
    # A cleaner way: get_token_sdr takes the map as an argument.

    return sp, tm, loaded_token_map # Return the map explicitly
