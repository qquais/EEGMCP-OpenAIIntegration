# BrainFlow-based EEG processing server
from flask import Flask, request, jsonify
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes
import os
import uuid
import numpy as np
import time

app = Flask(__name__)
UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def filter_edf_file(file_stream):
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}.edf")
    file_stream.save(filepath)

    params = BrainFlowInputParams()
    params.file = filepath
    board_id = BoardIds.SYNTHETIC_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(2)

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # Bandpass filter parameters
    start_freq = 0.5
    stop_freq = 40.0
    center_freq = (start_freq + stop_freq) / 2
    band_width = stop_freq - start_freq
    order = 4

    for ch in eeg_channels:
        DataFilter.perform_bandpass(
            data[ch],
            sampling_rate,
            center_freq,
            band_width,
            order,
            FilterTypes.BUTTERWORTH.value,
            0
        )

    filtered_data = {f'channel_{i+1}': data[ch].tolist() for i, ch in enumerate(eeg_channels)}
    os.remove(filepath)
    return filtered_data

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/filter-edf", methods=["POST"])
def filter_edf():
    try:
        file = request.files["file"]
        result = filter_edf_file(file)
        return jsonify({'filtered_data': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/features-edf", methods=["POST"])
def features_edf():
    try:
        file = request.files["file"]
        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f"{file_id}.edf")
        file.save(filepath)

        params = BrainFlowInputParams()
        params.file = filepath
        board_id = BoardIds.SYNTHETIC_BOARD.value

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(2)

        data = board.get_board_data()
        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        # ✅ Updated API: pass entire data, channel list, sampling_rate, and apply_filter=True
        bands, _ = DataFilter.get_avg_band_powers(
            data,
            eeg_channels,
            sampling_rate,
            apply_filter=True
        )

        band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        averaged_powers = dict(zip(band_names, bands))

        os.remove(filepath)
        return jsonify({"features": averaged_powers})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
