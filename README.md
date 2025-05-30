# HTM-LLM: An Experimental HTM-Based Large Language Model

This project is an experimental implementation of a Large Language Model (LLM) using Hierarchical Temporal Memory (HTM) principles. It demonstrates how Numenta's HTM theories can be applied to learn sequences from text data and generate responses in a chat-like interface.

## Overview

The system consists of:
* **SDR Utils (`sdr_utils.py`):** For tokenizing text and converting tokens into Sparse Distributed Representations (SDRs).
* **Spatial Pooler (`spatial_pooler.py`):** Learns stable SDRs for input tokens.
* **Temporal Memory (`temporal_memory.py`):** Learns sequences of SDRs from the Spatial Pooler and makes predictions.
* **Trainer (`htm_trainer.py`):** Manages the training loop for the SP and TM.
* **Persistence (`persistence.py`):** Saves and loads the trained model state.

## Prerequisites

* Python 3.x
* NumPy (`pip install numpy`)
    * (You may add other specific dependencies if you have them)

## Training the Model

1.  **Training Script:** The training process is handled by `run_htmllm.py`. This script initializes the HTM components (Spatial Pooler and Temporal Memory), processes training data, and saves the learned model state.
2.  **Training Data:** Currently, the training text is hardcoded within `run_htmllm.py`. You can modify the `raw_text` variable in this script to use your own dataset.
3.  **Running Training:**
    To start training, execute the following command in your terminal:
    ```bash
    python run_htmllm.py
    ```
4.  **Output:** The script will train the model. Upon completion, the trained model state (including SP permanences, TM segments, and token-SDR mappings) will be saved to a file named `htm_main_model_state.pkl`. Log files with detailed output from the training process will also be generated (e.g., `htm_output_log_YYYYMMDD_HHMMSS.txt`).

    **Note on Continuous Training:** By default, `run_htmllm.py` is set up to delete any existing `htm_main_model_state.pkl` file to ensure a fresh start for each training run. If you wish to continue training an existing model, you will need to comment out or remove the lines responsible for deleting this file in `run_htmllm.py`.

## Chatting with the Trained Model

1.  **Chat Script:** Once the model is trained and `htm_main_model_state.pkl` exists, you can interact with it using `chat.py`.
2.  **Running the Chat Interface:**
    Start the chat interface with the command:
    ```bash
    python chat.py
    ```
3.  **Interaction:**
    * The script will load the trained model from `htm_main_model_state.pkl`.
    * You can then type your input and press Enter.
    * The HTM model will attempt to generate a response based on its learned sequences.
    * To exit the chat, type `quit` and press Enter.
    * Chat interactions are logged to a file like `chat_interaction_log_YYYYMMDD_HHMMSS.txt`.

## Important: Verbosity

Please be aware that the current scripts, especially `temporal_memory.py` during its processing steps and `htm_trainer.py` during training, are **extremely verbose**. They print a lot of debug information to the console. This is intentional for understanding the model's internal state and behavior during development.

This detailed logging will make the console output quite noisy during both training and chat sessions.

---

Feel free to adapt and expand this as your project evolves!