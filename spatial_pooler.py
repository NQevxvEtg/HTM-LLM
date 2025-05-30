# spatial_pooler.py
import numpy as np
# Any other specific imports the SP class needs

class SpatialPooler:
    def __init__(self, input_size, num_columns, num_active_columns,
                 permanence_threshold=0.5, initial_permanence_std_dev=0.1,
                 permanence_increment=0.1, permanence_decrement=0.02, # Adjusted decrement
                 learning_rate=0.1,
                 global_inhibition=True, # Added
                 seed=None):
        self.input_size = input_size
        self.num_columns = num_columns
        self.num_active_columns = num_active_columns
        self.permanence_threshold = permanence_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.learning_rate = learning_rate # This isn't directly used in the Numenta formulation but scales increments
        self.global_inhibition = global_inhibition # If False, would need local inhibition logic

        if seed is not None:
            np.random.seed(seed) # Seed for SP's own random initializations

        # Initialize permanences
        self.permanences = np.random.normal(
            loc=self.permanence_threshold - 0.2, # Start a bit further from connected
            scale=initial_permanence_std_dev,
            size=(self.input_size, self.num_columns)
        )
        self.permanences = np.clip(self.permanences, 0.0, 1.0)

        # For tracking column activity (useful for boosting later)
        self.column_activations_count = np.zeros(self.num_columns)
        # Store raw overlaps for metrics
        self.last_overlaps = np.zeros(self.num_columns)


    def _get_connected_synapses(self):
        return self.permanences >= self.permanence_threshold

    def compute_overlap(self, input_sdr):
        connected_synapses = self._get_connected_synapses()
        overlaps = np.dot(input_sdr, connected_synapses)
        self.last_overlaps = overlaps.flatten() # Store for metrics
        return self.last_overlaps

    def activate_columns(self, overlaps):
        # Global k-winners-take-all inhibition
        active_columns_sdr = np.zeros(self.num_columns, dtype=int)
        if self.num_columns == 0: return active_columns_sdr

        if self.global_inhibition:
            if self.num_columns <= self.num_active_columns:
                top_indices = np.arange(self.num_columns)
            else:
                # Tie-breaking: add small random noise to overlaps before sorting
                noisy_overlaps = overlaps + np.random.uniform(-1e-6, 1e-6, size=overlaps.shape)
                top_indices = np.argsort(noisy_overlaps)[-self.num_active_columns:]
            active_columns_sdr[top_indices] = 1
        else:
            # Placeholder for local inhibition (more complex)
            # For now, if not global, activate all above a certain overlap threshold (not true HTM SP)
            # This part would need significant design for local inhibition.
            # For now, we'll assume global_inhibition=True
            pass
        return active_columns_sdr

    def learn(self, input_sdr, active_columns_sdr):
        active_input_indices = np.where(input_sdr == 1)[0]
        
        for col_idx in range(self.num_columns):
            is_active_column = (active_columns_sdr[col_idx] == 1)
            
            # For permanence update, consider a small neighborhood of inputs around connected synapses
            # For simplicity, we use all active_input_indices
            for input_idx in active_input_indices:
                current_perm = self.permanences[input_idx, col_idx]
                if is_active_column:
                    # Hebbian: If input is active and column is active, strengthen
                    self.permanences[input_idx, col_idx] = min(1.0, current_perm + self.permanence_increment)
                else:
                    # Anti-Hebbian-like: If input is active but column is not, slightly weaken
                    # This helps differentiate columns, but true Numenta SP handles this via boosting and duty cycles.
                    # A simpler approach is just to decrement if it was connected.
                    if current_perm >= self.permanence_threshold: # Only weaken if it was contributing
                         self.permanences[input_idx, col_idx] = max(0.0, current_perm - self.permanence_decrement)
        
        # Update column activation counts (for potential boosting later)
        self.column_activations_count += active_columns_sdr


    def process(self, input_sdr, learn=True):
        overlaps = self.compute_overlap(input_sdr)
        active_columns_sdr = self.activate_columns(overlaps)
        if learn:
            self.learn(input_sdr, active_columns_sdr)
        return active_columns_sdr, overlaps # Return overlaps for metrics
