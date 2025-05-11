# -*- coding: utf-8 -*-
"""
Robust Temporal Alignment Script for RGB and Infrared Videos using
DTW (on Motion Feature) and Event Matching Refinement.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import median_abs_deviation
import os # For checking file existence

# =============================================================================
# 1. Frame Preprocessing Function
# =============================================================================
def process_frame(frame, source_is_rgb):
    """
    Applies spatial transformations (cropping/resizing) to a single frame.

    !! IMPORTANT !!
    This function uses hardcoded dimensions based on the Wuxi dataset example.
    You WILL likely need to modify this for different video sources to ensure
    the RGB and IR frames cover the same approximate field of view and
    have the same dimensions after processing.

    Args:
        frame (np.ndarray): Input video frame (BGR format from OpenCV).
        source_is_rgb (bool): True if the frame is from the RGB source,
                              False if from the IR source.

    Returns:
        np.ndarray or None: Processed frame, or None if processing fails.
    """
    original_shape = frame.shape[:2] # (height, width)

    try:
        if source_is_rgb:
            # Example: RGB 1920x1080 -> Crop/Resize to match IR view
            if original_shape == (1080, 1920):
                # Resize first to approximate target dimensions
                rgb_resized = cv2.resize(frame, (704, 419), interpolation=cv2.INTER_AREA)
                # Crop to the specific overlapping region (example values)
                rgb_cropped = rgb_resized[:, 32:672] # Shape: (419, 640)
                return rgb_cropped
            else:
                # --- Placeholder for other RGB resolutions ---
                print(f"Warning: Untested RGB frame shape {original_shape}. Trying default resize/crop.")
                # Attempt a generic resize/crop (likely needs adjustment)
                target_h, target_w = 419, 640 # Target dimensions from IR example
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                return resized # Return resized, no specific crop known

        else: # Infrared Source
            # Example: IR 640x512 -> Crop to match RGB view
            if original_shape == (512, 640):
                ir_cropped = frame[66:485, :] # Shape: (419, 640)
                return ir_cropped
            else:
                 # --- Placeholder for other IR resolutions ---
                print(f"Warning: Untested IR frame shape {original_shape}. Trying default resize.")
                # Attempt a generic resize to match target (likely needs adjustment)
                target_h, target_w = 419, 640 # Target dimensions from RGB example
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                return resized

    except Exception as e:
        print(f"Error processing frame with shape {original_shape}: {e}")
        return None


# =============================================================================
# 2. Feature Extraction Function
# =============================================================================
def enhanced_feature_extraction(gray_frame, prev_gray_frame):
    """
    Calculates Motion Energy (center region) and Edge Density for a frame.

    Args:
        gray_frame (np.ndarray): Current frame (grayscale).
        prev_gray_frame (np.ndarray or None): Previous frame (grayscale).

    Returns:
        np.ndarray: Feature vector [motion_energy, edge_density].
    """
    motion_energy = 0.0
    edge_density = 0.0

    # --- Motion Energy (Frame Differencing in Center) ---
    if prev_gray_frame is not None:
        if gray_frame.shape == prev_gray_frame.shape and gray_frame.dtype == prev_gray_frame.dtype:
            try:
                diff = cv2.absdiff(gray_frame, prev_gray_frame)
                h, w = diff.shape
                # Calculate based on central 50% area to reduce edge noise impact
                center_y, center_x = h // 2, w // 2
                y_start, y_end = center_y // 2, center_y + center_y // 2
                x_start, x_end = center_x // 2, center_x + center_x // 2
                center_region = diff[y_start:y_end, x_start:x_end]

                if center_region.size > 0:
                    motion_energy = np.mean(center_region)
                else: # Fallback if center region is invalid
                    motion_energy = np.mean(diff)
            except Exception as e:
                print(f"Warning: Error calculating motion energy: {e}")
                motion_energy = 0.0 # Default on error
        else:
            # Shape/type mismatch, likely at sequence start or after skipped frame
            motion_energy = 0.0

    # --- Edge Density (Canny) ---
    try:
        # Using fixed thresholds, might need tuning for different sensors/lighting
        edges = cv2.Canny(gray_frame, 40, 120)
        edge_density = np.mean(edges) / 255.0 # Normalize to [0, 1]
    except Exception as e:
         print(f"Warning: Error calculating edge density: {e}")
         edge_density = 0.0 # Default on error


    return np.array([motion_energy, edge_density])


# =============================================================================
# 3. Video Reading and Feature Processing Function
# =============================================================================
def read_and_process_video(video_path, is_rgb):
    """
    Reads a video, processes frames, extracts features, and applies smoothing.

    Args:
        video_path (str): Path to the video file.
        is_rgb (bool): True if the video is RGB, False if IR.

    Returns:
        tuple: (list of processed frames, np.ndarray of smoothed features)
               Returns ([], np.array([])) if reading or processing fails.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return [], np.array([])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return [], np.array([])

    frames_processed = []
    features_raw = []
    prev_gray = None
    frame_count = 0
    processed_count = 0

    print(f"Processing video: {os.path.basename(video_path)}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1

        # 1. Preprocess frame (Resize/Crop)
        # processed = process_frame(frame, source_is_rgb=is_rgb)
        processed = frame
        if processed is None:
            print(f"Warning: Skipping frame {frame_count} due to processing error.")
            prev_gray = None # Reset history if frame is skipped
            continue

        # 2. Convert to Grayscale
        try:
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            elif len(processed.shape) == 2: # Already grayscale/single channel
                gray = processed
            else:
                raise ValueError(f"Unexpected processed frame shape: {processed.shape}")
        except Exception as e:
            print(f"Warning: Skipping frame {frame_count} due to grayscale conversion error: {e}")
            prev_gray = None
            continue

        if gray.size == 0:
            print(f"Warning: Skipping frame {frame_count} due to empty grayscale frame.")
            prev_gray = None
            continue

        # 3. Extract Features
        current_features = enhanced_feature_extraction(gray, prev_gray)
        features_raw.append(current_features)

        # 4. Store results and update history
        frames_processed.append(processed)
        prev_gray = gray
        processed_count += 1

    cap.release()
    print(f"Finished processing. Read {frame_count} frames, successfully processed {processed_count}.")

    if not features_raw:
        print("Error: No features were extracted.")
        return [], np.array([])

    # Set motion energy of the very first frame to 0
    if features_raw and len(features_raw[0]) > 0:
        features_raw[0][0] = 0.0

    features_array = np.array(features_raw)

    # 4. Smooth Features (Gaussian Filter)
    features_smoothed = np.array([])
    if features_array.ndim == 2 and features_array.shape[0] > 5: # Need enough data points
        try:
            sigma = 1.5 # Smoothing factor (adjust if needed)
            features_smoothed = np.zeros_like(features_array)
            for i in range(features_array.shape[1]): # Smooth each feature column
                features_smoothed[:, i] = gaussian_filter1d(features_array[:, i], sigma=sigma, mode='nearest')
            print(f"Applied Gaussian smoothing to features (sigma={sigma}).")
        except Exception as e:
            print(f"Warning: Could not apply feature smoothing: {e}")
            features_smoothed = features_array # Use raw features if smoothing fails
    else:
        print("Warning: Not enough data or incorrect dimensions for feature smoothing.")
        features_smoothed = features_array # Use raw features

    return frames_processed, features_smoothed


# =============================================================================
# 4. Dynamic Time Warping Function
# =============================================================================
def dynamic_time_warping(features1, features2):
    """
    Calculates the optimal path using Dynamic Time Warping (DTW).
    Assumes features are time series (N_frames, N_features).

    Args:
        features1 (np.ndarray): Feature sequence 1 (N, num_features).
        features2 (np.ndarray): Feature sequence 2 (M, num_features).

    Returns:
        list: DTW path [(idx1, idx2), ...], or empty list if failed.
    """
    n = features1.shape[0]
    m = features2.shape[0]

    if n == 0 or m == 0:
        print("Error: Empty feature sequence passed to DTW.")
        return []

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0 # Cost at origin is 0

    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate cost (Euclidean distance between feature vectors)
            cost = np.linalg.norm(features1[i - 1] - features2[j - 1])
            # Get minimum cost from neighbors
            last_min = min(dtw_matrix[i - 1, j],      # Insertion
                           dtw_matrix[i, j - 1],      # Deletion
                           dtw_matrix[i - 1, j - 1])  # Match
            dtw_matrix[i, j] = cost + last_min

    # Check if the end point is reachable
    if dtw_matrix[n, m] == np.inf:
        print("Error: DTW path could not reach the end cell (n, m). Alignment failed.")
        return []

    # Backtrack to find the optimal path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1)) # Store 0-based indices

        # Find the neighbor with the minimum cost
        possible_origins = []
        costs = []

        if i > 0: # Up
            possible_origins.append((i - 1, j))
            costs.append(dtw_matrix[i - 1, j])
        if j > 0: # Left
            possible_origins.append((i, j - 1))
            costs.append(dtw_matrix[i, j - 1])
        if i > 0 and j > 0: # Diagonal
            possible_origins.append((i - 1, j - 1))
            costs.append(dtw_matrix[i - 1, j - 1])

        if not costs: # Should not happen if end point was reached
             print("Error during DTW backtracking.")
             return []

        # Move to the cell with the minimum cost
        min_cost_idx = np.argmin(costs)
        i, j = possible_origins[min_cost_idx]


    # Path is reconstructed backwards, reverse it
    return path[::-1]

# =============================================================================
# 5. Key Event Detection Function
# =============================================================================
def find_key_events(features, feature_index=0, prominence_factor=0.15, min_interval=50):
    """
    Detects significant peaks and valleys in a feature time series.

    Args:
        features (np.ndarray): Feature array (N_frames, N_features).
        feature_index (int): Index of the feature column to analyze (e.g., 0 for motion).
        prominence_factor (float): Factor of the data range for peak prominence threshold.
        min_interval (int): Minimum number of frames between detected events.

    Returns:
        list: Sorted list of frame indices corresponding to key events.
    """
    if features.ndim != 2 or features.shape[1] <= feature_index:
        print(f"Error: Invalid features shape or feature index for event detection.")
        return []

    target_feature = features[:, feature_index]

    if target_feature.size < min_interval * 2: # Need sufficient data
        print("Warning: Not enough data points for robust peak finding.")
        return []

    data_range = np.ptp(target_feature) # Peak-to-peak range
    if data_range < 1e-7: # Avoid issues with flat signals
        print("Warning: Feature data range is near zero. Skipping event detection.")
        return []

    required_prominence = data_range * prominence_factor

    try:
        # Find peaks
        peaks, _ = find_peaks(target_feature, prominence=required_prominence, distance=min_interval)
        # Find valleys (on inverted signal)
        valleys, _ = find_peaks(-target_feature, prominence=required_prominence, distance=min_interval)
    except Exception as e:
        print(f"Error during scipy.signal.find_peaks: {e}")
        return []


    key_points = sorted(np.concatenate([peaks, valleys]).astype(int))
    print(f"Found {len(key_points)} key points (using Feature Index {feature_index}, Prominence Factor {prominence_factor:.2f}, Min Interval {min_interval}).")
    return list(key_points)

# =============================================================================
# 6. Main Alignment Function
# =============================================================================
def align_videos_robust(rgb_path, ir_path):
    """
    Performs robust temporal alignment between RGB and IR videos.

    Args:
        rgb_path (str): Path to the RGB video file.
        ir_path (str): Path to the Infrared video file.

    Returns:
        tuple: (aligned_rgb_frames, aligned_ir_frames, final_offset,
                rgb_features_all, ir_features_all)
               Returns ([], [], 0, None, None) on failure.
    """
    # --- 1. Read, Process, Extract Features ---
    rgb_frames, rgb_features_all = read_and_process_video(rgb_path, is_rgb=True)
    ir_frames, ir_features_all = read_and_process_video(ir_path, is_rgb=False)

    if not rgb_frames or not ir_frames or rgb_features_all.size == 0 or ir_features_all.size == 0:
        print("Error: Failed to process one or both videos. Cannot align.")
        # Return None for features if they couldn't be generated
        rgb_feat_out = None if rgb_features_all.size == 0 else rgb_features_all
        ir_feat_out = None if ir_features_all.size == 0 else ir_features_all
        return [], [], 0, rgb_feat_out, ir_feat_out

    # --- 2. Feature Normalization (Z-score) ---
    # Normalize both features for consistency, even if only motion used for DTW
    try:
        rgb_mean = np.mean(rgb_features_all, axis=0)
        rgb_std = np.std(rgb_features_all, axis=0)
        ir_mean = np.mean(ir_features_all, axis=0)
        ir_std = np.std(ir_features_all, axis=0)

        # Avoid division by zero for constant features
        rgb_std[rgb_std < 1e-7] = 1.0
        ir_std[ir_std < 1e-7] = 1.0

        rgb_feat_norm_all = (rgb_features_all - rgb_mean) / rgb_std
        ir_feat_norm_all = (ir_features_all - ir_mean) / ir_std
        print("Features normalized using Z-score.")
    except Exception as e:
        print(f"Error during feature normalization: {e}. Cannot align.")
        return [], [], 0, rgb_features_all, ir_features_all # Return unnormalized


    # --- 3. Initial Offset Estimation via DTW (on Motion Only) ---
    print("\nStep 1: Initial Offset Estimation using DTW (Motion Energy)...")
    # Select only the normalized motion feature (Index 0)
    rgb_motion_norm = rgb_feat_norm_all[:, 0:1] # Shape (N, 1)
    ir_motion_norm = ir_feat_norm_all[:, 0:1]   # Shape (M, 1)

    dtw_path = dynamic_time_warping(rgb_motion_norm, ir_motion_norm)

    if not dtw_path:
        print("Error: DTW failed. Cannot proceed with alignment.")
        return [], [], 0, rgb_features_all, ir_features_all # Return original features

    # Calculate median offset from the DTW path
    dtw_offsets = [j - i for i, j in dtw_path]
    if not dtw_offsets:
        main_offset_dtw = 0 # Should not happen if path exists, but safety fallback
        print("Warning: Could not calculate offsets from DTW path.")
    else:
        main_offset_dtw = int(np.median(dtw_offsets))
    print(f"Initial DTW median offset (Motion only): {main_offset_dtw} frames")

    # --- 4. Key Event Detection (on Motion Only) ---
    print("\nStep 2: Detecting Key Events (Motion Energy)...")
    # Parameters for event detection (tune if necessary)
    prominence_factor = 0.15
    min_interval = 50

    # Use the full normalized features array but specify index 0 for motion
    rgb_events = find_key_events(rgb_feat_norm_all, feature_index=0,
                                 prominence_factor=prominence_factor,
                                 min_interval=min_interval)
    ir_events = find_key_events(ir_feat_norm_all, feature_index=0,
                                prominence_factor=prominence_factor,
                                min_interval=min_interval)

    # Optional: Visualize detected events immediately for debugging
    # visualize_detected_events(rgb_feat_norm_all, ir_feat_norm_all, rgb_events, ir_events)

    # --- 5. Event Matching and Offset Refinement ---
    print("\nStep 3: Matching Events and Refining Offset...")
    event_pairs = []
    # Search window around the expected IR event index based on DTW offset
    search_window = 300 # Frames (tune if DTW is often very inaccurate)

    if rgb_events and ir_events:
        # Use a copy to potentially remove matched events (optional)
        ir_events_available = list(ir_events)
        for r_idx in rgb_events:
            expected_ir_idx = r_idx + main_offset_dtw
            potential_matches = []
            # Find the closest IR event within the search window
            for ir_idx in ir_events_available:
                if abs(ir_idx - expected_ir_idx) <= search_window:
                    # Store distance and index
                    potential_matches.append((abs(ir_idx - expected_ir_idx), ir_idx))

            if potential_matches:
                # Select the IR event closest in time within the window
                best_match_dist, best_match_idx = min(potential_matches, key=lambda x: x[0])
                event_pairs.append((r_idx, best_match_idx))
                # Optional: Prevent IR event from being matched multiple times
                # try: ir_events_available.remove(best_match_idx)
                # except ValueError: pass # Already removed

        print(f"Found {len(event_pairs)} potential event pairs (Search window: +/-{search_window} frames around DTW estimate).")
    else:
        print("Not enough events found in one or both videos to perform matching.")

    # --- Calculate final offset based on event pairs ---
    final_offset = main_offset_dtw # Default to DTW offset

    # Minimum number of pairs needed for statistical filtering
    MIN_EVENT_PAIRS_INITIAL = 5 # To trust event matching over DTW at all
    MIN_EVENT_PAIRS_FILTERED = 3 # To trust filtered median

    if len(event_pairs) >= MIN_EVENT_PAIRS_INITIAL:
        event_offsets = np.array([ir - rgb for rgb, ir in event_pairs])
        median_event_offset_initial = int(np.median(event_offsets))
        print(f"Initial median offset from {len(event_pairs)} event pairs: {median_event_offset_initial}")

        # --- Outlier Rejection using Median Absolute Deviation (MAD) ---
        if len(event_offsets) > 1: # Need >1 point for MAD
            try:
                mad = median_abs_deviation(event_offsets, scale='normal') # Approx std dev
                if mad > 1e-7: # Check if MAD is reasonably non-zero
                    # Factor determines strictness (e.g., 1.5-2.5)
                    outlier_factor = 1.5 # Tune if needed
                    outlier_threshold = outlier_factor * mad

                    filtered_event_pairs = []
                    filtered_event_offsets = []
                    for i, offset in enumerate(event_offsets):
                        # Keep pairs whose offset is close to the initial median
                        if abs(offset - median_event_offset_initial) <= outlier_threshold:
                            filtered_event_pairs.append(event_pairs[i])
                            filtered_event_offsets.append(offset)

                    num_outliers = len(event_offsets) - len(filtered_event_offsets)
                    print(f"Outlier rejection (MAD Filter): Removed {num_outliers} pairs (Threshold: {outlier_threshold:.2f}, MAD: {mad:.2f}). Kept {len(filtered_event_offsets)} pairs.")

                    # Use the median of filtered offsets if enough pairs remain
                    if len(filtered_event_offsets) >= MIN_EVENT_PAIRS_FILTERED:
                        final_offset = int(np.median(filtered_event_offsets))
                        print(f"Refined offset from {len(filtered_event_offsets)} filtered pairs: {final_offset}")
                    else:
                        # Fallback to initial median if too few pairs after filtering
                        final_offset = median_event_offset_initial
                        print(f"Warning: Too few pairs ({len(filtered_event_offsets)}) after outlier rejection. Using initial event median: {final_offset}")
                else:
                    # MAD is zero/tiny -> all offsets are nearly identical
                    final_offset = median_event_offset_initial
                    print("Event offsets highly consistent (MAD near zero). Using initial event median.")
            except Exception as e:
                print(f"Warning: Error during MAD calculation: {e}. Using initial event median.")
                final_offset = median_event_offset_initial

        elif len(event_offsets) == 1: # Only one pair found
             final_offset = event_offsets[0]
             print("Only one event pair found. Using its offset.")

        # Optional: Compare event offset with DTW offset
        if abs(final_offset - main_offset_dtw) > 100: # Arbitrary threshold for significant difference
             print(f"Note: Final event offset ({final_offset}) differs significantly from initial DTW offset ({main_offset_dtw}).")
        else:
             print(f"Final offset ({final_offset}) based on events is consistent with initial DTW offset ({main_offset_dtw}).")

    else: # Not enough initial pairs found
        print(f"Warning: Not enough initial event pairs found ({len(event_pairs)} < {MIN_EVENT_PAIRS_INITIAL}). Using DTW median offset as final result: {main_offset_dtw}")
        final_offset = main_offset_dtw

    print(f"\n--> Final Calculated Offset: {final_offset} frames")

    # --- 6. Apply Final Offset and Create Aligned Sequences ---
    print("\nStep 4: Applying offset and creating aligned sequences...")
    aligned_rgb_frames = []
    aligned_ir_frames = []
    num_aligned_frames = 0

    len_rgb = len(rgb_frames)
    len_ir = len(ir_frames)

    if final_offset >= 0:
        # IR starts later (or simultaneously if offset is 0)
        start_ir = final_offset
        start_rgb = 0
        # Check if overlap exists
        if start_ir < len_ir:
            # Calculate length of overlap
            num_aligned_frames = min(len_rgb - start_rgb, len_ir - start_ir)
            if num_aligned_frames > 0:
                aligned_rgb_frames = rgb_frames[start_rgb : start_rgb + num_aligned_frames]
                aligned_ir_frames = ir_frames[start_ir : start_ir + num_aligned_frames]
    else: # final_offset < 0
        # RGB starts later
        start_rgb = abs(final_offset)
        start_ir = 0
         # Check if overlap exists
        if start_rgb < len_rgb:
             # Calculate length of overlap
            num_aligned_frames = min(len_rgb - start_rgb, len_ir - start_ir)
            if num_aligned_frames > 0:
                aligned_rgb_frames = rgb_frames[start_rgb : start_rgb + num_aligned_frames]
                aligned_ir_frames = ir_frames[start_ir : start_ir + num_aligned_frames]

    if num_aligned_frames <= 0:
        print("Warning: Calculated offset results in zero overlapping frames.")
    else:
        print(f"Aligned sequence length: {num_aligned_frames} frames.")


    # Return aligned frames, offset, and the *original full* (unsmoothed) features for potential external use/visualization
    return aligned_rgb_frames, aligned_ir_frames, final_offset, rgb_features_all, ir_features_all


# =============================================================================
# 7. Visualization Functions
# =============================================================================
def visualize_alignment(rgb_features, ir_features, offset, feature_names=['Motion Energy', 'Edge Density']):
    """
    Visualizes the alignment of feature sequences.

    Args:
        rgb_features (np.ndarray): RGB features (N_frames, N_features).
        ir_features (np.ndarray): IR features (M_frames, N_features).
        offset (int): Calculated frame offset (ir_index = rgb_index + offset).
        feature_names (list): Names of the features for plot labels.
    """
    if rgb_features is None or ir_features is None or rgb_features.size == 0 or ir_features.size == 0:
        print("Skipping feature alignment visualization due to missing features.")
        return

    num_features = rgb_features.shape[1]
    if num_features != ir_features.shape[1]:
         print("Warning: Feature dimensions mismatch between RGB and IR. Cannot visualize alignment.")
         return
    if len(feature_names) != num_features:
        feature_names = [f"Feature {i+1}" for i in range(num_features)] # Default names

    plt.figure(figsize=(15, 4 * num_features))
    plt.suptitle(f"Feature Alignment Visualization (Offset = {offset})", fontsize=14)

    rgb_time = np.arange(len(rgb_features))
    ir_time = np.arange(len(ir_features))
    # Calculate aligned time axis for IR relative to RGB's axis
    # ir_frame[t] aligns with rgb_frame[t - offset]
    # So, if RGB time is 'x', the corresponding IR time is 'x + offset'
    # Plotting IR feature at 'x' means taking ir_feature[x + offset]
    # Alternatively, plot ir_feature[t] at time point 't - offset' on RGB's axis
    ir_time_aligned_on_rgb_axis = ir_time - offset

    for i in range(num_features):
        ax = plt.subplot(num_features, 1, i + 1)
        ax.plot(rgb_time, rgb_features[:, i], label=f'RGB {feature_names[i]}', alpha=0.8, linewidth=1.5)
        ax.plot(ir_time_aligned_on_rgb_axis, ir_features[:, i], label=f'IR {feature_names[i]} (Aligned)',
                linestyle='--', alpha=0.8, linewidth=1.5)
        ax.set_title(f"Alignment: {feature_names[i]}")
        ax.set_xlabel("Frame Number (Relative to RGB Start)")
        ax.set_ylabel("Feature Value")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()

def display_sample_frames(rgb_list, ir_list, offset, num_samples=5):
    """
    Displays side-by-side samples of aligned RGB and IR frames.

    Args:
        rgb_list (list): List of aligned RGB frames.
        ir_list (list): List of aligned IR frames.
        offset (int): The calculated final offset.
        num_samples (int): Number of sample pairs to display.
    """
    if not rgb_list or not ir_list:
        print("No aligned frames available to display.")
        return

    num_aligned = len(rgb_list)
    if num_aligned == 0:
        print("Aligned frame lists are empty.")
        return

    if num_aligned < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {num_aligned} aligned frames exist. Displaying all.")
        num_samples = num_aligned
        sample_indices = np.arange(num_aligned)
    else:
        # Select evenly spaced samples across the aligned sequence
        sample_indices = np.linspace(0, num_aligned - 1, num_samples, dtype=int)

    print(f"\nDisplaying {len(sample_indices)} sample aligned frame pairs...")
    plt.figure(figsize=(10, num_samples * 2.5)) # Adjusted figure size
    plt.suptitle(f"Aligned Frame Samples (Offset = {offset})", fontsize=14)

    for i, aligned_idx in enumerate(sample_indices):
        rgb_frame = rgb_list[aligned_idx]
        ir_frame = ir_list[aligned_idx]

        # Calculate original frame indices for context
        if offset >= 0:
            original_rgb_idx = aligned_idx
            original_ir_idx = aligned_idx + offset
        else:
            original_rgb_idx = aligned_idx + abs(offset)
            original_ir_idx = aligned_idx

        # --- Prepare frames for display ---
        # Ensure IR is BGR for stacking (convert if grayscale)
        if len(ir_frame.shape) == 2:
            ir_display = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
        elif len(ir_frame.shape) == 3 and ir_frame.shape[2] == 1:
            ir_display = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
        else:
            ir_display = ir_frame

        # Ensure frames have same height for hstack (should be guaranteed by process_frame)
        if rgb_frame.shape[0] != ir_display.shape[0]:
             # If height mismatch occurs unexpectedly, try resizing IR to match RGB
            print(f"Warning: Height mismatch in sample {i}. Resizing IR.")
            target_h = rgb_frame.shape[0]
            scale = target_h / ir_display.shape[0]
            target_w = int(ir_display.shape[1] * scale)
            if target_w > 0:
                 ir_display = cv2.resize(ir_display, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else: continue # Skip if resize fails

        # Combine frames side-by-side
        try:
            combined_frame = np.hstack((rgb_frame, ir_display))
        except ValueError as e:
            print(f"Error combining frames for sample {i}: {e}. Shapes: RGB={rgb_frame.shape}, IR={ir_display.shape}")
            continue

        # Display using Matplotlib (convert BGR to RGB)
        ax = plt.subplot(num_samples, 1, i + 1)
        ax.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
        title_str = (f"Sample {i+1} (Aligned Idx: {aligned_idx})\n"
                     f"Orig RGB Frame: {original_rgb_idx}, Orig IR Frame: {original_ir_idx}")
        ax.set_title(title_str, fontsize=9)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


# =============================================================================
# 8. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    print("Starting RGB-IR Video Temporal Alignment Script...")

    # --- USER INPUT: Specify video paths ---
    rgb_video_path = 'processed_smoked1_rgb.mp4' # REPLACE WITH YOUR RGB VIDEO PATH
    ir_video_path = 'processed_smoked1_ir_final.mp4'   # REPLACE WITH YOUR IR VIDEO PATH
    # ---

    # --- Run Alignment ---
    aligned_rgb, aligned_ir, final_offset, rgb_features, ir_features = align_videos_robust(
        rgb_video_path, ir_video_path
    )

    # --- Display Results ---
    print("\n--- Alignment Results ---")
    if aligned_rgb and aligned_ir:
        print(f"Alignment Successful!")
        print(f"Final Calculated Offset: {final_offset} frames")
        print(f"   (Interpretation: IR frame index = RGB frame index + {final_offset})")
        print(f"Length of Aligned Sequences: {len(aligned_rgb)} frames")

        # --- Visualize Feature Alignment ---
        # Pass the features returned by align_videos_robust
        visualize_alignment(rgb_features, ir_features, final_offset)

        # --- Display Sample Aligned Frames ---
        display_sample_frames(aligned_rgb, aligned_ir, final_offset, num_samples=5)

    else:
        print("Alignment Failed.")
        print(f"Final Offset Calculation Resulted In: {final_offset}") # Show offset even if alignment failed

    print("\nAlignment script finished.")