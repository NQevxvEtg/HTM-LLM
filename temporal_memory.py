# temporal_memory.py
import numpy as np

class TemporalMemory:
    def __init__(self, num_columns, cells_per_column=1,
                 activation_threshold=3,
                 initial_permanence=0.21,
                 connected_permanence=0.50,
                 learning_threshold=1, # Keeping the lower threshold we discussed
                 permanence_increment=0.10,
                 permanence_decrement=0.05,
                 max_synapses_per_segment=32,
                 max_segments_per_cell=128,
                 seed=None):

        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.num_cells = self.num_columns * self.cells_per_column

        self.activation_threshold = activation_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.learning_threshold = learning_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.max_synapses_per_segment = max_synapses_per_segment
        self.max_segments_per_cell = max_segments_per_cell

        if seed is not None:
            np.random.seed(seed)

        self.active_cells = np.zeros(self.num_cells, dtype=bool)
        self.predictive_cells = np.zeros(self.num_cells, dtype=int)
        self.winner_cells = np.zeros(self.num_cells, dtype=bool)
        
        self.segments = [[] for _ in range(self.num_cells)]
        # --- DEBUG PRINTS ---
        print(f"    DEBUG TM __init__: Inside TemporalMemory __init__.")
        print(f"    DEBUG TM __init__: self.num_columns = {self.num_columns}")
        print(f"    DEBUG TM __init__: self.cells_per_column = {self.cells_per_column}")
        print(f"    DEBUG TM __init__: self.num_cells = {self.num_cells}")
        if self.num_cells > 0:
            print(f"    DEBUG TM __init__: Initialized self.segments for {self.num_cells} cells. Example: len(self.segments[0]) = {len(self.segments[0])}")
        else:
            print(f"    DEBUG TM __init__: self.num_cells is 0, critical error likely.")
        print(f"    DEBUG TM __init__: self.max_segments_per_cell = {self.max_segments_per_cell}")
        # --- END DEBUG PRINTS ---
        self.prev_active_cells = np.zeros(self.num_cells, dtype=bool)
        self.prev_winner_cells = np.zeros(self.num_cells, dtype=bool)

        print(f"TM Initialized: {self.num_columns} cols, {self.cells_per_column} cell/col.")
        print(f"  Activation Threshold: {self.activation_threshold}, Learning Threshold: {self.learning_threshold}")

# In temporal_memory.py

    def _activate_segments(self, cell_idx_list, active_cells_from_context, is_matching_phase):
        # ... (threshold calculation remains the same) ...
        threshold = self.learning_threshold if is_matching_phase else self.activation_threshold

        if is_matching_phase:
            active_segments_for_cells = [[] for _ in cell_idx_list] # For learning phase
        else:
            # CHANGE: For prediction phase, this will store the strength of the prediction, or 0 if not predictive.
            predictive_strength_for_cells = [0] * len(cell_idx_list)

        for i, cell_idx in enumerate(cell_idx_list):
            if not is_matching_phase:
                max_strength_for_this_cell = 0
                best_segment_idx_for_this_cell = -1 # For debug print

            for seg_idx, segment in enumerate(self.segments[cell_idx]):
                num_active_connected_synapses = 0
                for presynaptic_cell_idx, permanence in segment['synapses'].items():
                    if permanence >= self.connected_permanence and \
                       active_cells_from_context[presynaptic_cell_idx]:
                        num_active_connected_synapses += 1
                
                if is_matching_phase:
                    if num_active_connected_synapses >= threshold:
                        active_segments_for_cells[i].append(segment) # type: ignore
                else: # Prediction phase
                    if num_active_connected_synapses >= threshold:
                        if num_active_connected_synapses > max_strength_for_this_cell: # type: ignore
                            max_strength_for_this_cell = num_active_connected_synapses # type: ignore
                            best_segment_idx_for_this_cell = seg_idx # type: ignore
            
            if not is_matching_phase and max_strength_for_this_cell > 0: # type: ignore
                predictive_strength_for_cells[i] = max_strength_for_this_cell # type: ignore
                # This print remains very useful for debugging
                print(f"    !!!! Cell [{cell_idx}] is PREDICTIVE for t+1 via seg_idx [{best_segment_idx_for_this_cell}] with {max_strength_for_this_cell} active synapses !!!!") # type: ignore
        
        if is_matching_phase:
            return active_segments_for_cells
        else:
            return predictive_strength_for_cells

    def _adapt_segment(self, segment, active_cells_from_prev_step, learning_rate_factor=1.0):
        for presynaptic_cell_idx, permanence in list(segment['synapses'].items()):
            new_permanence = permanence
            if active_cells_from_prev_step[presynaptic_cell_idx]:
                new_permanence = min(1.0, permanence + self.permanence_increment * learning_rate_factor)
                if new_permanence != permanence : 
                    if new_permanence >= self.connected_permanence and permanence < self.connected_permanence:
                         print(f"      Synapse [{presynaptic_cell_idx}] on segment: {permanence:.2f} -> {new_permanence:.2f} (REINFORCED & NOW CONNECTED)")
                    elif new_permanence >= self.connected_permanence:
                         print(f"      Synapse [{presynaptic_cell_idx}] on segment: {permanence:.2f} -> {new_permanence:.2f} (REINFORCED & REMAINS CONNECTED)")
                    else:
                        print(f"      Synapse [{presynaptic_cell_idx}] on segment: {permanence:.2f} -> {new_permanence:.2f} (REINFORCED)")
            else: 
                new_permanence = max(0.0, permanence - self.permanence_decrement * learning_rate_factor)
                if new_permanence != permanence and permanence >= self.connected_permanence: 
                    print(f"      Synapse [{presynaptic_cell_idx}] on segment: {permanence:.2f} -> {new_permanence:.2f} (WEAKENED, was connected)")
            segment['synapses'][presynaptic_cell_idx] = new_permanence
        
        num_current_synapses = len(segment['synapses'])
        if num_current_synapses < self.max_synapses_per_segment:
            potential_new_synapses = 0
            candidate_presynaptic_cells = np.where(active_cells_from_prev_step == True)[0]
            num_to_add = self.max_synapses_per_segment - num_current_synapses # Corrected variable name
            
            if candidate_presynaptic_cells.size > 0:
                np.random.shuffle(candidate_presynaptic_cells) 
                for presynaptic_cell_idx in candidate_presynaptic_cells:
                    if potential_new_synapses >= num_to_add: # CORRECTED: num_to_add
                        break
                    if presynaptic_cell_idx not in segment['synapses'] or \
                       segment['synapses'].get(presynaptic_cell_idx, 0) < self.initial_permanence: 
                        segment['synapses'][presynaptic_cell_idx] = self.initial_permanence
                        potential_new_synapses +=1
                        print(f"      NEW/RESET Synapse formed on segment for prev active cell [{presynaptic_cell_idx}] with initial perm: {self.initial_permanence:.2f}")
        return segment

    def _get_or_create_learning_segment(self, cell_idx, active_cells_from_prev_step):
        best_segment_to_reinforce = None
        max_raw_overlap = -1
        
        if self.segments[cell_idx]:
            for segment_candidate in self.segments[cell_idx]:
                current_raw_overlap = 0
                for presynaptic_cell_idx in segment_candidate['synapses']:
                    if active_cells_from_prev_step[presynaptic_cell_idx]:
                        current_raw_overlap += 1
                
                if current_raw_overlap > max_raw_overlap:
                    max_raw_overlap = current_raw_overlap
                    best_segment_to_reinforce = segment_candidate # Assign the segment object
        
        current_segment_count_for_cell = len(self.segments[cell_idx])
        print(f"    DEBUG _get_or_create: Cell [{cell_idx}], Current segment count: {current_segment_count_for_cell}, max_segments_per_cell: {self.max_segments_per_cell}")

        if best_segment_to_reinforce is not None and max_raw_overlap >= self.learning_threshold:
            print(f"    Cell [{cell_idx}]: Found existing segment to reinforce (raw overlap: {max_raw_overlap}, synapses: {len(best_segment_to_reinforce['synapses'])}).")
        elif current_segment_count_for_cell < self.max_segments_per_cell:
            print(f"    Cell [{cell_idx}]: No suitable existing segment found (max raw overlap {max_raw_overlap} < thresh {self.learning_threshold}). Creating NEW segment (context: {np.where(active_cells_from_prev_step)[0]})")
            new_segment = {'synapses': {}, 'is_sequence_segment': True} 
            self.segments[cell_idx].append(new_segment)
            best_segment_to_reinforce = new_segment
            self._adapt_segment(best_segment_to_reinforce, active_cells_from_prev_step, learning_rate_factor=1.0) 
        else: 
            print(f"    Cell [{cell_idx}]: No suitable existing segment (max raw overlap {max_raw_overlap} < thresh {self.learning_threshold}) and no capacity to create new one.")
            best_segment_to_reinforce = None 
        
        print(f"    DEBUG _get_or_create: Condition for creating new (based on logic): {current_segment_count_for_cell < self.max_segments_per_cell and (best_segment_to_reinforce is None or (isinstance(best_segment_to_reinforce, dict) and len(best_segment_to_reinforce['synapses']) == 0 and max_raw_overlap < self.learning_threshold)) }") # Adjusted debug print
        return best_segment_to_reinforce

# In temporal_memory.py

    def process(self, sp_active_columns_sdr, learn=True):
        # CHANGE: Adjust print statement for previous predictive_cells if it stores strengths
        print(f"\nTM Process Input (SP columns): {np.where(sp_active_columns_sdr == 1)[0]}")
        # self.predictive_cells now stores strengths; np.where(self.predictive_cells > 0) gets predictive indices
        print(f"  Previous predictive_cells (absolute cell indices with strength > 0): {np.where(self.predictive_cells > 0)[0]}")

        current_step_active_cells = np.zeros(self.num_cells, dtype=bool)
        current_step_winner_cells = np.zeros(self.num_cells, dtype=bool)
        
        bursting_columns_indices = []
        winner_cell_indices_for_log = []

        active_column_indices_from_sp = np.where(sp_active_columns_sdr == 1)[0]

        for col_idx in active_column_indices_from_sp:
            column_was_predicted = False
            # CHANGE: Store (abs_cell_idx, prediction_strength)
            predicted_cells_with_strength_in_this_column = [] 

            for cell_in_col_offset in range(self.cells_per_column):
                abs_cell_idx = col_idx * self.cells_per_column + cell_in_col_offset
                # self.predictive_cells[abs_cell_idx] now holds the strength from the previous step's prediction
                prediction_strength = self.predictive_cells[abs_cell_idx]
                
                if prediction_strength > 0: # Cell was predictive if its strength is positive
                    predicted_cells_with_strength_in_this_column.append((abs_cell_idx, prediction_strength))
                    column_was_predicted = True
            
            if column_was_predicted:
                if predicted_cells_with_strength_in_this_column:
                    # Sort by strength (descending) to find the cell with the strongest prediction
                    predicted_cells_with_strength_in_this_column.sort(key=lambda x: x[1], reverse=True)
                    chosen_winner_cell_abs_idx = predicted_cells_with_strength_in_this_column[0][0]
                    
                    current_step_active_cells[chosen_winner_cell_abs_idx] = True
                    current_step_winner_cells[chosen_winner_cell_abs_idx] = True
                    winner_cell_indices_for_log.append(chosen_winner_cell_abs_idx)
                # Else: This case (column_was_predicted but list is empty) shouldn't happen if logic is correct.
            else: # Bursting
                bursting_columns_indices.append(col_idx)
                for cell_in_col_offset in range(self.cells_per_column):
                    abs_cell_idx = col_idx * self.cells_per_column + cell_in_col_offset
                    current_step_active_cells[abs_cell_idx] = True
                    
        self.active_cells[:] = current_step_active_cells[:]
        self.winner_cells[:] = current_step_winner_cells[:]
        
        print(f"  Active TM Cells (absolute cell indices): {np.where(self.active_cells)[0]}")
        if winner_cell_indices_for_log: print(f"  Winner Cells (correctly predicted, absolute indices): {winner_cell_indices_for_log}")
        if bursting_columns_indices: print(f"  Bursting Columns (indices): {bursting_columns_indices} (all their cells are active)")
        
        # --- Phase 2: Adapt Segments and Compute Predictive Cells for t+1 ---
        # CHANGE: self.predictive_cells will be reset to 0s (as it stores strengths)
        self.predictive_cells.fill(0)

        # ... (Learning phase: _adapt_segment, _get_or_create_learning_segment logic remains largely the same) ...
        # The context for learning (self.prev_active_cells) is still boolean.
        if learn:
            print(f"    DEBUG TM Learn (Before Check): self.prev_active_cells (abs cell indices) = {np.where(self.prev_active_cells)[0]}")
            if np.any(self.prev_active_cells):
                print(f"  Learning Phase - Context (prev_active_cells - abs cell indices): {np.where(self.prev_active_cells)[0]}")
                for abs_cell_idx_learning in np.where(self.active_cells)[0]: 
                    print(f"  Attempting to learn for active cell (abs index) [{abs_cell_idx_learning}]")
                    segment_to_learn_on = self._get_or_create_learning_segment(abs_cell_idx_learning, self.prev_active_cells)
                    if segment_to_learn_on:
                        self._adapt_segment(segment_to_learn_on, self.prev_active_cells)
                    else:
                        print(f"    No segment to learn on for cell [{abs_cell_idx_learning}] (either no match or no capacity).")
            else:
                print("  Learning Phase - No previous active cells, skipping segment adaptation.")


        if np.any(self.active_cells):
            print(f"  Predicting for t+1. Context for prediction (current active_cells - abs cell indices): {np.where(self.active_cells)[0]}")
            cells_to_check_prediction_for = list(range(self.num_cells))
            
            # CHANGE: This now returns strengths
            prediction_strengths_for_next_step = self._activate_segments(
                cells_to_check_prediction_for,
                self.active_cells, 
                is_matching_phase=False
            )
            # Assign these strengths to self.predictive_cells
            for i, abs_cell_idx_pred_check in enumerate(cells_to_check_prediction_for):
                self.predictive_cells[abs_cell_idx_pred_check] = prediction_strengths_for_next_step[i]
        
        # CHANGE: Adjust print for outputting predictive_cells
        print(f"  Outputting predictive_cells for t+1 (abs cell indices with strength > 0): {np.where(self.predictive_cells > 0)[0]}")

        self.prev_active_cells[:] = self.active_cells[:] 
        self.prev_winner_cells[:] = self.winner_cells[:]
        print(f"    DEBUG TM Learn (After Update): self.prev_active_cells now (abs cell indices) = {np.where(self.prev_active_cells)[0]}")
        
        # The return value remains self.predictive_cells.astype(int).
        # If predictive_cells stores strengths, .astype(int) is fine.
        # The trainer will interpret non-zero as predictive.
        return self.predictive_cells.astype(int)