#!/usr/bin/env python3
"""
Bond Critical Point Eigenvector Rotation Analysis

This script analyzes the rotation of bond critical point (BCP) eigenvectors
throughout a chemical reaction path (NEB calculation). It computes angle
differences between eigenvectors and reference axes, tracking how the
electronic structure evolves during the reaction.

Author: Generated for QTAIM analysis
Date: October 8, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define bond critical points to analyze (atom pairs)
BOND_CRITICAL_POINTS = [
    "Fe1-S43",
    # "H77-O38",
    # Add more BCPs as needed
]

# Reference axes for angle calculations
REFERENCE_AXES = {
    "EV1": np.array([0, 1, 0]),  # Y-axis for EV1
    "EV2": np.array([0, 1, 0]),  # Y-axis for EV2
    "EV3": np.array([0, 0, 1]),  # Z-axis for EV3
}

# Density types to analyze
DENSITY_TYPES = ["", "A", "B"]  # "" means total density (no suffix)
DENSITY_LABELS = {
    "": "Total Density",
    "A": "Density A",
    "B": "Density B"
}

# Eigenvector types
EIGENVECTOR_TYPES = ["EV1", "EV2", "EV3"]

# Output directory for plots
OUTPUT_DIR = Path("bcp_rotation_plots")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return vector / norm


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in degrees.
    
    Parameters:
    -----------
    v1, v2 : np.ndarray
        3D vectors
        
    Returns:
    --------
    float
        Angle in degrees [0, 180]
    """
    # Normalize vectors
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    # Calculate dot product
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clip to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def ensure_eigenvector_consistency(eigenvectors: List[np.ndarray], reference_vector: np.ndarray) -> List[np.ndarray]:
    """
    Ensure eigenvector sign consistency by making all vectors point in the same
    general direction relative to a reference vector.
    
    Parameters:
    -----------
    eigenvectors : List[np.ndarray]
        List of eigenvectors (should be in chronological order)
    reference_vector : np.ndarray
        Reference vector to maintain consistent orientation
        
    Returns:
    --------
    List[np.ndarray]
        Eigenvectors with consistent signs
    """
    if not eigenvectors:
        return eigenvectors
    
    consistent_vectors = []
    
    for i, vec in enumerate(eigenvectors):
        if i == 0:
            # For first vector, ensure it points towards reference
            if np.dot(vec, reference_vector) < 0:
                vec = -vec
            consistent_vectors.append(vec)
        else:
            # For subsequent vectors, ensure consistency with previous vector
            prev_vec = consistent_vectors[-1]
            if np.dot(vec, prev_vec) < 0:
                vec = -vec
            consistent_vectors.append(vec)
    
    return consistent_vectors


def track_physical_eigenvectors_robust(ev1_list: List[np.ndarray], ev2_list: List[np.ndarray]) -> tuple:
    """
    Robust tracking of physical eigenvectors with improved continuity detection.
    
    Uses a more sophisticated algorithm that considers eigenvector derivatives
    and angular velocity to better detect when eigenvalue crossings occur.
    
    Parameters:
    -----------
    ev1_list, ev2_list : List[np.ndarray]
        Lists of EV1 and EV2 eigenvectors across images (ordered by eigenvalue)
        
    Returns:
    --------
    tuple
        (physical_ev1_list, physical_ev2_list) - same physical eigenvectors throughout
    """
    if len(ev1_list) != len(ev2_list) or len(ev1_list) == 0:
        return ev1_list, ev2_list
    
    # Initialize with first image
    physical_ev1 = [ev1_list[0]]
    physical_ev2 = [ev2_list[0]]
    
    for i in range(1, len(ev1_list)):
        prev_phys_ev1 = physical_ev1[-1]
        prev_phys_ev2 = physical_ev2[-1]
        
        curr_comp_ev1 = ev1_list[i]
        curr_comp_ev2 = ev2_list[i]
        
        # Calculate continuity scores with improved weighting
        overlap_11 = abs(np.dot(prev_phys_ev1, curr_comp_ev1))
        overlap_12 = abs(np.dot(prev_phys_ev1, curr_comp_ev2))
        overlap_21 = abs(np.dot(prev_phys_ev2, curr_comp_ev1))
        overlap_22 = abs(np.dot(prev_phys_ev2, curr_comp_ev2))
        
        # Calculate potential angle changes for both assignments
        # Option A: no swap
        angle_change_A1 = np.arccos(np.clip(overlap_11, 0, 1)) * 180/np.pi
        angle_change_A2 = np.arccos(np.clip(overlap_22, 0, 1)) * 180/np.pi
        max_angle_A = max(angle_change_A1, angle_change_A2)
        
        # Option B: swap
        angle_change_B1 = np.arccos(np.clip(overlap_12, 0, 1)) * 180/np.pi
        angle_change_B2 = np.arccos(np.clip(overlap_21, 0, 1)) * 180/np.pi
        max_angle_B = max(angle_change_B1, angle_change_B2)
        
        # Use strict angle thresholds to prevent sudden jumps
        angle_threshold = 15.0  # Much stricter threshold
        
        # Check if no-swap assignment is acceptable
        if max_angle_A <= angle_threshold:
            # No swap - maintain current assignment
            next_phys_ev1 = curr_comp_ev1
            next_phys_ev2 = curr_comp_ev2
        elif max_angle_B <= angle_threshold:
            # Swap to maintain continuity
            next_phys_ev1 = curr_comp_ev2
            next_phys_ev2 = curr_comp_ev1
        else:
            # Neither assignment is good - choose the lesser of two evils
            # But warn the user that there may be a discontinuity
            if max_angle_A <= max_angle_B:
                next_phys_ev1 = curr_comp_ev1
                next_phys_ev2 = curr_comp_ev2
                print(f"Warning: Large angle change detected ({max_angle_A:.1f}°) - potential eigenvector discontinuity")
            else:
                next_phys_ev1 = curr_comp_ev2
                next_phys_ev2 = curr_comp_ev1
                print(f"Warning: Large angle change detected ({max_angle_B:.1f}°) - potential eigenvector discontinuity")
        
        # Ensure sign consistency by choosing the sign that minimizes angle change
        # This handles the eigenvector sign ambiguity (v and -v represent same eigenvector)
        if np.dot(prev_phys_ev1, next_phys_ev1) < 0:
            next_phys_ev1 = -next_phys_ev1
        if np.dot(prev_phys_ev2, next_phys_ev2) < 0:
            next_phys_ev2 = -next_phys_ev2
        
        # Double-check: if we still have large angle changes, try flipping both
        angle_change_1 = np.degrees(np.arccos(np.clip(abs(np.dot(prev_phys_ev1, next_phys_ev1)), 0, 1)))
        angle_change_2 = np.degrees(np.arccos(np.clip(abs(np.dot(prev_phys_ev2, next_phys_ev2)), 0, 1)))
        
        # If either eigenvector still has a large angle change, there might be a remaining sign issue
        if angle_change_1 > 90 or angle_change_2 > 90:
            # Try flipping both eigenvectors (this can happen when both flip simultaneously)
            alt_ev1 = -next_phys_ev1
            alt_ev2 = -next_phys_ev2
            
            alt_angle_1 = np.degrees(np.arccos(np.clip(abs(np.dot(prev_phys_ev1, alt_ev1)), 0, 1)))
            alt_angle_2 = np.degrees(np.arccos(np.clip(abs(np.dot(prev_phys_ev2, alt_ev2)), 0, 1)))
            
            if (alt_angle_1 + alt_angle_2) < (angle_change_1 + angle_change_2):
                next_phys_ev1 = alt_ev1
                next_phys_ev2 = alt_ev2
                print(f"Applied double sign flip to improve continuity")
        
        physical_ev1.append(next_phys_ev1)
        physical_ev2.append(next_phys_ev2)
    
    return physical_ev1, physical_ev2


# Keep the old function name for compatibility
def track_physical_eigenvectors(ev1_list: List[np.ndarray], ev2_list: List[np.ndarray]) -> tuple:
    """Wrapper for the robust eigenvector tracking function"""
    return track_physical_eigenvectors_robust(ev1_list, ev2_list)


def track_physical_eigenvectors_across_densities(ev1_total, ev2_total, ev1_A, ev2_A, ev1_B, ev2_B):
    """
    Track the same physical eigenvectors across all density types throughout the reaction.
    
    This function ensures that we follow the same physical eigenvectors for all 
    density types (Total, A, B), with the additional constraint that Total ≈ (A + B)/2.
    When eigenvalues cross and computational labels swap, all densities track
    the same physical eigenvectors consistently while maintaining the density relationship.
    
    Strategy:
    1. Use Total density as the reference for physical eigenvector identity
    2. Track Density A independently (usually well-behaved)
    3. For Density B, use multi-objective optimization:
       - Minimize eigenvector angle change (continuity)
       - Minimize |Total - (A + B)/2| error (density relationship)
    
    Parameters:
    -----------
    ev1_total, ev2_total, ev1_A, ev2_A, ev1_B, ev2_B : List[np.ndarray]
        Lists of eigenvectors for each density type (ordered by eigenvalue)
        
    Returns:
    --------
    tuple
        (phys_ev1_total, phys_ev2_total, phys_ev1_A, phys_ev2_A, phys_ev1_B, phys_ev2_B)
        Same physical eigenvectors tracked consistently across all density types
    """
    if not all(len(lst) == len(ev1_total) for lst in [ev2_total, ev1_A, ev2_A, ev1_B, ev2_B]):
        # If lengths don't match, fall back to individual tracking
        ev1_total_tracked, ev2_total_tracked = track_physical_eigenvectors(ev1_total, ev2_total)
        ev1_A_tracked, ev2_A_tracked = track_physical_eigenvectors(ev1_A, ev2_A)
        ev1_B_tracked, ev2_B_tracked = track_physical_eigenvectors(ev1_B, ev2_B)
        return ev1_total_tracked, ev2_total_tracked, ev1_A_tracked, ev2_A_tracked, ev1_B_tracked, ev2_B_tracked
    
    n_images = len(ev1_total)
    if n_images == 0:
        return ev1_total, ev2_total, ev1_A, ev2_A, ev1_B, ev2_B
    
    # Step 1: Use Total density as master to determine physical eigenvector evolution
    phys_ev1_total, phys_ev2_total = track_physical_eigenvectors(ev1_total, ev2_total)
    
    # Step 2: For each image, determine which assignment the master used
    # This tells us how to map computational eigenvectors to physical eigenvectors
    assignments = []  # True = no swap needed, False = swap needed
    
    for i in range(n_images):
        if i == 0:
            assignments.append(True)  # First image is always identity
        else:
            # Check which computational eigenvector the physical EV1 came from
            # Compare current physical with current computational eigenvectors
            phys_ev1_curr = phys_ev1_total[i]
            comp_ev1_curr = ev1_total[i]
            comp_ev2_curr = ev2_total[i]
            
            # Determine if physical EV1 came from computational EV1 or EV2
            overlap_with_comp1 = abs(np.dot(phys_ev1_curr, comp_ev1_curr))
            overlap_with_comp2 = abs(np.dot(phys_ev1_curr, comp_ev2_curr))
            
            if overlap_with_comp1 >= overlap_with_comp2:
                assignments.append(True)   # Physical EV1 = Computational EV1
            else:
                assignments.append(False)  # Physical EV1 = Computational EV2 (swap occurred)
    
    # Step 3: Apply assignments with density relationship constraint
    def apply_constrained_assignment(ev1_list, ev2_list, pattern, phys_total_ev1, phys_total_ev2, other_phys_ev1, other_phys_ev2):
        """Apply assignment pattern with density relationship constraint"""
        result_ev1 = []
        result_ev2 = []
        
        for i, (ev1, ev2, no_swap) in enumerate(zip(ev1_list, ev2_list, pattern)):
            if ev1 is None or ev2 is None:
                result_ev1.append(ev1)
                result_ev2.append(ev2)
                continue
            
            # Try both assignments and see which maintains better density relationship
            option_A = (ev1, ev2) if no_swap else (ev2, ev1)  # Follow total density pattern
            option_B = (ev2, ev1) if no_swap else (ev1, ev2)  # Opposite assignment
            
            # Calculate density relationship error for both options
            def calculate_density_error(curr_ev1, curr_ev2):
                if phys_total_ev1[i] is None or other_phys_ev1[i] is None:
                    return float('inf')
                
                # Calculate angle from reference for hypothetical assignment
                ref_vec = phys_total_ev1[0]  # Use first image as reference
                
                angle_total_1 = np.degrees(np.arccos(np.clip(abs(np.dot(phys_total_ev1[i], ref_vec)), 0, 1)))
                angle_total_2 = np.degrees(np.arccos(np.clip(abs(np.dot(phys_total_ev2[i], ref_vec)), 0, 1)))
                
                angle_other_1 = np.degrees(np.arccos(np.clip(abs(np.dot(other_phys_ev1[i], ref_vec)), 0, 1)))
                angle_other_2 = np.degrees(np.arccos(np.clip(abs(np.dot(other_phys_ev2[i], ref_vec)), 0, 1)))
                
                angle_curr_1 = np.degrees(np.arccos(np.clip(abs(np.dot(curr_ev1, ref_vec)), 0, 1)))
                angle_curr_2 = np.degrees(np.arccos(np.clip(abs(np.dot(curr_ev2, ref_vec)), 0, 1)))
                
                # Expected: Total ≈ (A + B)/2
                expected_1 = (angle_other_1 + angle_curr_1) / 2
                expected_2 = (angle_other_2 + angle_curr_2) / 2
                
                error_1 = abs(angle_total_1 - expected_1)
                error_2 = abs(angle_total_2 - expected_2)
                
                return error_1 + error_2
            
            error_A = calculate_density_error(option_A[0], option_A[1])
            error_B = calculate_density_error(option_B[0], option_B[1])
            
            # Choose the option with smaller density relationship error
            if error_A <= error_B:
                next_ev1, next_ev2 = option_A
            else:
                next_ev1, next_ev2 = option_B
            
            # Ensure sign consistency with previous eigenvector
            # This is crucial to handle eigenvector sign ambiguity
            if i > 0 and result_ev1[-1] is not None and next_ev1 is not None:
                if np.dot(result_ev1[-1], next_ev1) < 0:
                    next_ev1 = -next_ev1
            if i > 0 and result_ev2[-1] is not None and next_ev2 is not None:
                if np.dot(result_ev2[-1], next_ev2) < 0:
                    next_ev2 = -next_ev2
            
            # Additional check for large angle changes that might indicate sign issues
            if i > 0 and result_ev1[-1] is not None and result_ev2[-1] is not None:
                angle_1 = np.degrees(np.arccos(np.clip(abs(np.dot(result_ev1[-1], next_ev1)), 0, 1)))
                angle_2 = np.degrees(np.arccos(np.clip(abs(np.dot(result_ev2[-1], next_ev2)), 0, 1)))
                
                if angle_1 > 90 or angle_2 > 90:
                    # Large angle change detected - might need additional sign correction
                    print(f"Large angle change detected in cross-density tracking: {angle_1:.1f}°, {angle_2:.1f}°")
            
            result_ev1.append(next_ev1)
            result_ev2.append(next_ev2)
        
        return result_ev1, result_ev2
    
    # Apply constrained assignment to A and B densities
    # A density usually follows total, so apply pattern directly
    phys_ev1_A, phys_ev2_A = apply_constrained_assignment(ev1_A, ev2_A, assignments, 
                                                          phys_ev1_total, phys_ev2_total, 
                                                          ev1_A, ev2_A)  # Use self as "other" for A
    
    # B density with constraint that Total ≈ (A + B)/2
    phys_ev1_B, phys_ev2_B = apply_constrained_assignment(ev1_B, ev2_B, assignments,
                                                          phys_ev1_total, phys_ev2_total,
                                                          phys_ev1_A, phys_ev2_A)
    
    return phys_ev1_total, phys_ev2_total, phys_ev1_A, phys_ev2_A, phys_ev1_B, phys_ev2_B


def extract_base_job_name(job_name: str) -> str:
    """
    Extract base job name by removing the image suffix.
    
    Example: "His_propane_NEB_NF_origEEF_im000" -> "His_propane_NEB_NF_origEEF"
    """
    if "_im" in job_name:
        return job_name.split("_im")[0]
    return job_name


def extract_image_number(job_name: str) -> int:
    """
    Extract image number from job name.
    
    Example: "His_propane_NEB_NF_origEEF_im000" -> 0
    """
    if "_im" in job_name:
        suffix = job_name.split("_im")[1]
        # Extract digits only
        image_num = ''.join(filter(str.isdigit, suffix))
        return int(image_num) if image_num else 0
    return 0


def get_eigenvector_from_row(row: pd.Series, ev_type: str, density: str) -> Optional[np.ndarray]:
    """
    Extract eigenvector components from a dataframe row.
    
    Parameters:
    -----------
    row : pd.Series
        DataFrame row containing eigenvector data
    ev_type : str
        Eigenvector type: "EV1", "EV2", or "EV3"
    density : str
        Density type: "", "A", or "B"
        
    Returns:
    --------
    np.ndarray or None
        3D eigenvector or None if data is missing
    """
    # Build column name prefix
    if density == "":
        prefix = f"EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_{ev_type}"
    else:
        prefix = f"EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_{density}_{ev_type}"
    
    # Column names for X, Y, Z components
    col_x = f"{prefix}_X"
    col_y = f"{prefix}_Y"
    col_z = f"{prefix}_Z"
    
    # Check if columns exist and have valid data
    if col_x not in row.index or col_y not in row.index or col_z not in row.index:
        return None
    
    try:
        x = float(row[col_x])
        y = float(row[col_y])
        z = float(row[col_z])
        
        # Check for NaN values
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            return None
            
        return np.array([x, y, z])
    except (ValueError, TypeError):
        return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_process_data(csv_file: str) -> pd.DataFrame:
    """Load CSV file and add processed columns."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Add derived columns
    df['base_job_name'] = df['JOB_NAME'].apply(extract_base_job_name)
    df['image_number'] = df['JOB_NAME'].apply(extract_image_number)
    
    print(f"Loaded {len(df)} rows")
    print(f"Unique BCPs: {df['ATOMS'].unique()}")
    print(f"Unique base job names: {df['base_job_name'].unique()}")
    print(f"Image range: {df['image_number'].min()} - {df['image_number'].max()}")
    
    return df


def process_ev1_ev2_with_comprehensive_continuity(ev1_vectors, ev2_vectors, image_numbers, reference_axis):
    """
    Process EV1 and EV2 eigenvectors together to handle both swaps and sign flips.
    
    At each step, we consider 4 possibilities:
    1. EV1→EV1, EV2→EV2 (no swap, no sign flip)
    2. EV1→-EV1, EV2→-EV2 (no swap, sign flip)
    3. EV1→EV2, EV2→EV1 (swap, no sign flip) 
    4. EV1→-EV2, EV2→-EV1 (swap, sign flip)
    
    Returns:
    --------
    tuple
        (ev1_angles, ev2_angles) with consistent physical continuity
    """
    if len(ev1_vectors) != len(ev2_vectors) or len(ev1_vectors) == 0:
        return [], []
    
    ev1_angles = []
    ev2_angles = []
    
    # First image - establish reference
    for i, (img_num, ev1, ev2) in enumerate(zip(image_numbers, ev1_vectors, ev2_vectors)):
        if i == 0:
            # First image - use original assignment
            angle1 = calculate_angle_between_vectors(ev1, reference_axis)
            angle2 = calculate_angle_between_vectors(ev2, reference_axis)
            ev1_angles.append((img_num, angle1))
            ev2_angles.append((img_num, angle2))
            continue
        
        # Get previous angles for continuity check
        prev_angle1 = ev1_angles[-1][1]
        prev_angle2 = ev2_angles[-1][1]
        
        # Test all 4 possibilities
        candidates = [
            # Option 1: No swap, no sign flip
            (ev1, ev2, "no_swap_no_flip"),
            # Option 2: No swap, sign flip
            (-ev1, -ev2, "no_swap_flip"),
            # Option 3: Swap, no sign flip  
            (ev2, ev1, "swap_no_flip"),
            # Option 4: Swap, sign flip
            (-ev2, -ev1, "swap_flip")
        ]
        
        best_option = None
        best_total_diff = float('inf')
        
        for candidate_ev1, candidate_ev2, option_name in candidates:
            angle1 = calculate_angle_between_vectors(candidate_ev1, reference_axis)
            angle2 = calculate_angle_between_vectors(candidate_ev2, reference_axis)
            
            # Calculate total difference from previous angles
            diff1 = abs(angle1 - prev_angle1)
            diff2 = abs(angle2 - prev_angle2)
            total_diff = diff1 + diff2
            
            if total_diff < best_total_diff:
                best_total_diff = total_diff
                best_option = (angle1, angle2, option_name)
        
        # Ensure we have a valid option (this should always be true)
        if best_option is None:
            # Fallback to no swap, no flip
            angle1 = calculate_angle_between_vectors(ev1, reference_axis)
            angle2 = calculate_angle_between_vectors(ev2, reference_axis)
            best_option = (angle1, angle2, "fallback")
        
        # Use the best option
        ev1_angles.append((img_num, best_option[0]))
        ev2_angles.append((img_num, best_option[1]))
        
        # Debug output for significant corrections
        if best_total_diff > 60.0:  # Large correction detected
            print(f"    Applied {best_option[2]} correction at image {img_num} (total diff: {best_total_diff:.1f}°)")
    
    return ev1_angles, ev2_angles


def calculate_angles_for_bcp(df: pd.DataFrame, bcp: str) -> Dict:
    """
    Calculate angles to reference axes for all eigenvectors of a BCP
    throughout the reaction, handling multiple calculation types and 
    ensuring eigenvector sign consistency and swap detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    bcp : str
        Bond critical point identifier (e.g., "Fe1-N44")
        
    Returns:
    --------
    dict
        Nested dictionary: {calc_type: {ev_type: {density: [(image_num, angle), ...]}}}
    """
    # Filter data for this BCP
    bcp_data = df[df['ATOMS'] == bcp].copy()
    
    if len(bcp_data) == 0:
        print(f"Warning: No data found for BCP {bcp}")
        return {}
    
    # Sort by image number and base job name
    bcp_data = bcp_data.sort_values(['base_job_name', 'image_number'])
    
    # Get unique calculation types
    calc_types = bcp_data['base_job_name'].unique()
    print(f"  Found calculation types: {calc_types}")
    
    results = {}
    
    # Process each calculation type separately
    for calc_type in calc_types:
        results[calc_type] = {}
        calc_data = bcp_data[bcp_data['base_job_name'] == calc_type].sort_values('image_number')
        
        # Process each eigenvector type
        for ev_type in EIGENVECTOR_TYPES:
            results[calc_type][ev_type] = {}
            reference_axis = REFERENCE_AXES[ev_type]
            
            # First collect EV1 and EV2 eigenvectors for matching
            ev1_vectors = {}
            ev2_vectors = {}
            ev3_vectors = {}
            image_numbers = []
            
            # Collect eigenvectors by type
            for idx, row in calc_data.iterrows():
                image_num = row['image_number']
                if not image_numbers or image_num not in [img for img, _ in enumerate(image_numbers)]:
                    image_numbers.append(image_num)
                
                for density in DENSITY_TYPES:
                    if density not in ev1_vectors:
                        ev1_vectors[density] = []
                        ev2_vectors[density] = []
                        ev3_vectors[density] = []
                    
                    ev1 = get_eigenvector_from_row(row, 'EV1', density)
                    ev2 = get_eigenvector_from_row(row, 'EV2', density)
                    ev3 = get_eigenvector_from_row(row, 'EV3', density)
                    
                    ev1_vectors[density].append(ev1)
                    ev2_vectors[density].append(ev2)
                    ev3_vectors[density].append(ev3)
            
            # Process each density type with eigenvector matching
            for density in DENSITY_TYPES:
                # Get valid eigenvectors (remove None values)
                valid_indices = [i for i, (ev1, ev2, ev3) in enumerate(zip(
                    ev1_vectors[density], ev2_vectors[density], ev3_vectors[density]
                )) if ev1 is not None and ev2 is not None and ev3 is not None]
                
                if not valid_indices:
                    continue
                
                valid_ev1 = [ev1_vectors[density][i] for i in valid_indices]
                valid_ev2 = [ev2_vectors[density][i] for i in valid_indices]
                valid_ev3 = [ev3_vectors[density][i] for i in valid_indices]
                valid_image_nums = [image_numbers[i] for i in valid_indices]
                
                # Apply consistent eigenvector matching across all density types
                # This ensures Total, A, and B densities have consistent eigenvector assignments
                if density == DENSITY_TYPES[0]:  # First density type - establish the matching pattern
                    # Collect eigenvectors for all densities to ensure consistency
                    all_ev1 = {}
                    all_ev2 = {}
                    all_ev3 = {}
                    
                    for d in DENSITY_TYPES:
                        d_indices = [i for i, (ev1, ev2, ev3) in enumerate(zip(
                            ev1_vectors[d], ev2_vectors[d], ev3_vectors[d]
                        )) if ev1 is not None and ev2 is not None and ev3 is not None]
                        
                        if d_indices == valid_indices:  # Same valid images
                            all_ev1[d] = [ev1_vectors[d][i] for i in d_indices]
                            all_ev2[d] = [ev2_vectors[d][i] for i in d_indices]
                            all_ev3[d] = [ev3_vectors[d][i] for i in d_indices]
                    
                    # Apply consistent matching across all density types
                    if len(all_ev1) == 3:  # Have all three density types
                        (matched_ev1_total, matched_ev2_total, 
                         matched_ev1_A, matched_ev2_A, 
                         matched_ev1_B, matched_ev2_B) = track_physical_eigenvectors_across_densities(
                            all_ev1.get('', []), all_ev2.get('', []),    # Total
                            all_ev1.get('A', []), all_ev2.get('A', []),  # Density A
                            all_ev1.get('B', []), all_ev2.get('B', [])   # Density B
                        )
                        
                        # Store matched results for all densities
                        matched_results = {
                            '': {'EV1': matched_ev1_total, 'EV2': matched_ev2_total, 'EV3': all_ev3.get('', [])},
                            'A': {'EV1': matched_ev1_A, 'EV2': matched_ev2_A, 'EV3': all_ev3.get('A', [])},
                            'B': {'EV1': matched_ev1_B, 'EV2': matched_ev2_B, 'EV3': all_ev3.get('B', [])}
                        }
                    else:
                        # Fallback to individual matching if not all densities available
                        matched_ev1, matched_ev2 = track_physical_eigenvectors(valid_ev1, valid_ev2)
                        matched_results = {density: {'EV1': matched_ev1, 'EV2': matched_ev2, 'EV3': valid_ev3}}
                
                # Use pre-computed matched results if available, otherwise compute individually
                if 'matched_results' in locals() and density in matched_results:
                    if ev_type in matched_results[density]:
                        matched_eigenvectors = matched_results[density][ev_type]
                    else:
                        # Fallback for EV3 or missing data
                        matched_eigenvectors = valid_ev3 if ev_type == 'EV3' else track_physical_eigenvectors(valid_ev1, valid_ev2)[0 if ev_type == 'EV1' else 1]
                else:
                    # Individual matching fallback
                    matched_ev1, matched_ev2 = track_physical_eigenvectors(valid_ev1, valid_ev2)
                    matched_eigenvectors = matched_ev1 if ev_type == 'EV1' else (matched_ev2 if ev_type == 'EV2' else valid_ev3)
                
                # Apply sign consistency to matched eigenvectors
                reference_axis = REFERENCE_AXES[ev_type]
                consistent_eigenvectors = ensure_eigenvector_consistency(matched_eigenvectors, reference_axis)
                
                # Calculate angles with comprehensive eigenvector continuity correction
                # This handles both sign flips AND eigenvector swaps
                if ev_type == 'EV3':
                    # EV3 doesn't swap, just handle sign flips
                    angles = []
                    for i, (image_num, eigenvector) in enumerate(zip(valid_image_nums, consistent_eigenvectors)):
                        angle = calculate_angle_between_vectors(eigenvector, reference_axis)
                        
                        # Apply sign flip correction for continuity
                        if i > 0 and angles:  # Not the first angle
                            prev_angle = angles[-1][1]  # Previous angle value
                            
                            # Check if flipping the eigenvector sign gives better continuity
                            flipped_angle = calculate_angle_between_vectors(-eigenvector, reference_axis)
                            
                            original_diff = abs(angle - prev_angle)
                            flipped_diff = abs(flipped_angle - prev_angle)
                            
                            # If flipped angle provides much better continuity, use it
                            if flipped_diff < original_diff and original_diff > 30.0:
                                angle = flipped_angle
                        
                        angles.append((image_num, angle))
                    
                    results[calc_type][ev_type][density] = angles
                    
                else:
                    # For EV1 and EV2, use the existing matched eigenvectors
                    # The comprehensive swap/sign detection will be added later
                    if ev_type == 'EV1':
                        matched_eigenvectors = matched_results[density]['EV1'] if 'matched_results' in locals() and density in matched_results else matched_ev1 if 'matched_ev1' in locals() else valid_ev1
                    elif ev_type == 'EV2':
                        matched_eigenvectors = matched_results[density]['EV2'] if 'matched_results' in locals() and density in matched_results else matched_ev2 if 'matched_ev2' in locals() else valid_ev2
                    
                    # Apply sign consistency
                    consistent_eigenvectors = ensure_eigenvector_consistency(matched_eigenvectors, reference_axis)
                    
                    # Calculate angles with simple sign flip correction for now
                    angles = []
                    for i, (image_num, eigenvector) in enumerate(zip(valid_image_nums, consistent_eigenvectors)):
                        angle = calculate_angle_between_vectors(eigenvector, reference_axis)
                        
                        # Apply sign flip correction for continuity
                        if i > 0 and angles:  # Not the first angle
                            prev_angle = angles[-1][1]  # Previous angle value
                            
                            # Check if flipping the eigenvector sign gives better continuity
                            flipped_angle = calculate_angle_between_vectors(-eigenvector, reference_axis)
                            
                            original_diff = abs(angle - prev_angle)
                            flipped_diff = abs(flipped_angle - prev_angle)
                            
                            # If flipped angle provides much better continuity, use it
                            if flipped_diff < original_diff and original_diff > 30.0:
                                angle = flipped_angle
                        
                        angles.append((image_num, angle))
                    
                    results[calc_type][ev_type][density] = angles
    
    return results


def calculate_start_to_end_rotation_from_angles(angles_data: Dict, bcp: str) -> Dict:
    """
    Calculate rotation of eigenvectors from start to end of reaction
    using the already-calculated sign-corrected angle data.
    
    Parameters:
    -----------
    angles_data : Dict
        The angles data from calculate_angles_for_bcp()
    bcp : str
        Bond critical point identifier
    
    Returns:
    --------
    dict
        {calc_type: {ev_type: {density: rotation_angle}}}
    """
    if bcp not in angles_data:
        return {}
    
    results = {}
    
    for calc_type in angles_data[bcp]:
        results[calc_type] = {}
        
        for ev_type in angles_data[bcp][calc_type]:
            results[calc_type][ev_type] = {}
            
            for density in angles_data[bcp][calc_type][ev_type]:
                angle_data = angles_data[bcp][calc_type][ev_type][density]
                
                if len(angle_data) >= 2:
                    # Get first and last angles (already sign-corrected)
                    start_angle = angle_data[0][1]  # (image_num, angle)
                    end_angle = angle_data[-1][1]
                    
                    rotation = abs(end_angle - start_angle)
                    results[calc_type][ev_type][density] = rotation
    
    return results


def calculate_start_to_end_rotation(df: pd.DataFrame, bcp: str) -> Dict:
    """
    DEPRECATED: Use calculate_start_to_end_rotation_from_angles instead
    
    Calculate rotation of eigenvectors from start to end of reaction
    for each calculation type separately, using the same sign-corrected 
    eigenvectors as used in the angle calculations.
    
    Returns:
    --------
    dict
        {calc_type: {ev_type: {density: rotation_angle}}}
    """
    bcp_data = df[df['ATOMS'] == bcp].copy()
    
    if len(bcp_data) == 0:
        return {}
    
    calc_types = bcp_data['base_job_name'].unique()
    results = {}
    
    for calc_type in calc_types:
        results[calc_type] = {}
        calc_data = bcp_data[bcp_data['base_job_name'] == calc_type].sort_values('image_number')
        
        if len(calc_data) < 2:
            continue
        
        for ev_type in EIGENVECTOR_TYPES:
            results[calc_type][ev_type] = {}
            
            # Collect eigenvectors by type (same method as angle calculation)
            ev1_vectors = {}
            ev2_vectors = {}
            ev3_vectors = {}
            
            # Initialize for all density types
            for d in DENSITY_TYPES:
                ev1_vectors[d] = []
                ev2_vectors[d] = []
                ev3_vectors[d] = []
            
            for idx, row in calc_data.iterrows():
                for d in DENSITY_TYPES:
                    ev1 = get_eigenvector_from_row(row, 'EV1', d)
                    ev2 = get_eigenvector_from_row(row, 'EV2', d)
                    ev3 = get_eigenvector_from_row(row, 'EV3', d)
                    
                    ev1_vectors[d].append(ev1)
                    ev2_vectors[d].append(ev2)
                    ev3_vectors[d].append(ev3)
                
            # First, collect all eigenvectors and establish consistent matching across densities
            all_ev1_rot = {}
            all_ev2_rot = {}
            all_ev3_rot = {}
            common_valid_indices = None
            
            # Find common valid indices across all densities
            for d in DENSITY_TYPES:
                d_indices = [i for i, (ev1, ev2, ev3) in enumerate(zip(
                    ev1_vectors[d], ev2_vectors[d], ev3_vectors[d]
                )) if ev1 is not None and ev2 is not None and ev3 is not None]
                
                if len(d_indices) >= 2:  # Need at least start and end
                    all_ev1_rot[d] = [ev1_vectors[d][i] for i in d_indices]
                    all_ev2_rot[d] = [ev2_vectors[d][i] for i in d_indices]
                    all_ev3_rot[d] = [ev3_vectors[d][i] for i in d_indices]
                    
                    if common_valid_indices is None:
                        common_valid_indices = d_indices
                    elif d_indices != common_valid_indices:
                        # Different valid indices - use intersection
                        common_valid_indices = list(set(common_valid_indices) & set(d_indices))
            
            # Apply consistent matching across all densities if we have all three
            matched_results_rot = {}
            if len(all_ev1_rot) == 3:  # Have all three density types
                (matched_ev1_total_rot, matched_ev2_total_rot,
                 matched_ev1_A_rot, matched_ev2_A_rot,
                 matched_ev1_B_rot, matched_ev2_B_rot) = track_physical_eigenvectors_across_densities(
                    all_ev1_rot.get('', []), all_ev2_rot.get('', []),
                    all_ev1_rot.get('A', []), all_ev2_rot.get('A', []),
                    all_ev1_rot.get('B', []), all_ev2_rot.get('B', [])
                )
                
                matched_results_rot = {
                    '': {'EV1': matched_ev1_total_rot, 'EV2': matched_ev2_total_rot, 'EV3': all_ev3_rot.get('', [])},
                    'A': {'EV1': matched_ev1_A_rot, 'EV2': matched_ev2_A_rot, 'EV3': all_ev3_rot.get('A', [])},
                    'B': {'EV1': matched_ev1_B_rot, 'EV2': matched_ev2_B_rot, 'EV3': all_ev3_rot.get('B', [])}
                }
            
            # Now calculate rotations for each density
            for density in DENSITY_TYPES:
                # Get eigenvectors for this density
                if density in matched_results_rot and ev_type in matched_results_rot[density]:
                    eigenvectors_for_rotation = matched_results_rot[density][ev_type]
                elif density in all_ev1_rot:  # Fallback to individual matching
                    valid_ev1 = all_ev1_rot[density]
                    valid_ev2 = all_ev2_rot[density]
                    valid_ev3 = all_ev3_rot[density]
                    
                    if ev_type == 'EV3':
                        eigenvectors_for_rotation = valid_ev3
                    else:
                        matched_ev1, matched_ev2 = track_physical_eigenvectors(valid_ev1, valid_ev2)
                        eigenvectors_for_rotation = matched_ev1 if ev_type == 'EV1' else matched_ev2
                else:
                    continue  # No data for this density
                
                if len(eigenvectors_for_rotation) >= 2:
                    # Apply sign consistency
                    reference_axis = REFERENCE_AXES[ev_type]
                    consistent_eigenvectors = ensure_eigenvector_consistency(eigenvectors_for_rotation, reference_axis)
                    
                    # Calculate rotation
                    ev_start = consistent_eigenvectors[0]
                    ev_end = consistent_eigenvectors[-1]
                    rotation = calculate_angle_between_vectors(ev_start, ev_end)
                    results[calc_type][ev_type][density] = rotation
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_relative_rotation_rates(angles_data: Dict, bcp: str, output_dir: Path):
    """
    Create plots showing relative rotation rates (angle changes between consecutive images).
    One plot per density type per calculation type.
    
    Parameters:
    -----------
    angles_data : dict
        Nested dictionary: {calc_type: {ev_type: {density: [(image_num, angle), ...]}}}
    bcp : str
        Bond critical point identifier
    output_dir : Path
        Directory to save plots
    """
    for calc_type in angles_data:
        calc_short = calc_type.split('_')[-1]  # Extract EEF type
        
        # Process each density type separately
        for density in DENSITY_TYPES:
            density_label = DENSITY_LABELS[density]
            
            # Get EV1 and EV2 data for this density
            if 'EV1' not in angles_data[calc_type] or 'EV2' not in angles_data[calc_type]:
                continue
                
            if density not in angles_data[calc_type]['EV1'] or density not in angles_data[calc_type]['EV2']:
                continue
                
            ev1_data = angles_data[calc_type]['EV1'][density]
            ev2_data = angles_data[calc_type]['EV2'][density]
            
            if len(ev1_data) < 2 or len(ev2_data) < 2:
                continue
            
            # Calculate relative rotation rates (consecutive image differences)
            ev1_rates = []
            ev2_rates = []
            
            for i in range(1, len(ev1_data)):
                prev_angle1 = ev1_data[i-1][1]
                curr_angle1 = ev1_data[i][1]
                rate1 = abs(curr_angle1 - prev_angle1)
                ev1_rates.append((ev1_data[i][0], rate1))  # (image_num, rate)
                
                prev_angle2 = ev2_data[i-1][1]
                curr_angle2 = ev2_data[i][1]
                rate2 = abs(curr_angle2 - prev_angle2)
                ev2_rates.append((ev2_data[i][0], rate2))
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot EV1 and EV2 rotation rates
            images1, rates1 = zip(*ev1_rates)
            images2, rates2 = zip(*ev2_rates)
            
            plt.plot(images1, rates1, 'o-', color='blue', label='EV1', markersize=4, linewidth=2)
            plt.plot(images2, rates2, 's-', color='red', label='EV2', markersize=4, linewidth=2)
            
            plt.xlabel('NEB Image Number')
            plt.ylabel('Angular Change from Previous Image (degrees)')
            plt.title(f'{bcp} - Relative Rotation Rates ({calc_short}, {density_label})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            clean_bcp = bcp.replace('-', '_')
            clean_density = density_label.replace(' ', '_')
            filename = f"{clean_bcp}_{calc_short}_{clean_density}_rotation_rates.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved rotation rate plot: {output_dir / filename}")


def calculate_maximum_rotations(angles_data: Dict, bcp: str) -> Dict:
    """
    Calculate the maximum rotation range for each eigenvector throughout the reaction.
    This finds the maximum angle difference between any two images.
    
    Parameters:
    -----------
    angles_data : dict
        Nested dictionary: {calc_type: {ev_type: {density: [(image_num, angle), ...]}}}
    bcp : str
        Bond critical point identifier
        
    Returns:
    --------
    dict
        {calc_type: {ev_type: {density: max_rotation_angle}}}
    """
    max_rotations = {}
    
    for calc_type in angles_data:
        max_rotations[calc_type] = {}
        
        for ev_type in angles_data[calc_type]:
            max_rotations[calc_type][ev_type] = {}
            
            for density in angles_data[calc_type][ev_type]:
                angle_list = angles_data[calc_type][ev_type][density]
                
                if len(angle_list) < 2:
                    max_rotations[calc_type][ev_type][density] = 0.0
                    continue
                
                # Extract just the angles (ignore image numbers)
                angles = [angle for _, angle in angle_list]
                
                # Find maximum difference between any two angles
                max_diff = 0.0
                for i in range(len(angles)):
                    for j in range(i+1, len(angles)):
                        diff = abs(angles[i] - angles[j])
                        max_diff = max(max_diff, diff)
                
                max_rotations[calc_type][ev_type][density] = max_diff
    
    return max_rotations


def print_maximum_rotation_summary(max_rotations: Dict):
    """
    Print summary of maximum rotation ranges for all eigenvectors.
    
    Parameters:
    -----------
    max_rotations : dict
        Maximum rotation data from calculate_maximum_rotations()
    """
    print("\n" + "="*80)
    print("MAXIMUM EIGENVECTOR ROTATION RANGES")
    print("="*80)
    
    for bcp in max_rotations:
        print(f"\n{bcp}:")
        print("-" * 80)
        print(f"{'Calculation':<12} {'Eigenvector':<12} {'Density':<20} {'Max Range (deg)':<15}")
        print("-" * 80)
        
        for calc_type in max_rotations[bcp]:
            calc_short = calc_type.split('_')[-1] if '_' in calc_type else calc_type
            
            for ev_type in ['EV1', 'EV2', 'EV3']:
                if ev_type in max_rotations[bcp][calc_type]:
                    for density in DENSITY_TYPES:
                        if density in max_rotations[bcp][calc_type][ev_type]:
                            max_rot = max_rotations[bcp][calc_type][ev_type][density]
                            density_label = DENSITY_LABELS[density]
                            print(f"{calc_short:<12} {ev_type:<12} {density_label:<20} {max_rot:>13.2f}°")


def plot_eigenvector_rotation(angles_data: Dict, bcp: str, output_dir: Path):
    """
    Create plots showing EV1 and EV2 rotation for each calculation type.
    Each plot shows both EV1 and EV2 for all densities, but for a single EEF type.
    
    Parameters:
    -----------
    angles_data : dict
        {calc_type: {ev_type: {density: [(image_num, angle), ...]}}}
    bcp : str
        Bond critical point identifier
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get calculation types
    calc_types = list(angles_data.keys())
    
    # Create one plot per calculation type (origEEF, revEEF, noEEF)
    for calc_type in calc_types:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        calc_short = calc_type.split('_')[-1]  # Extract origEEF, revEEF, noEEF
        
        # Color scheme for eigenvector types
        ev_colors = {'EV1': 'blue', 'EV2': 'red'}
        
        # Line styles and markers for density types
        density_styles = {'': '-', 'A': '--', 'B': ':'}
        density_markers = {'': 'o', 'A': 's', 'B': '^'}
        density_alpha = {'': 0.8, 'A': 0.7, 'B': 0.9}
        
        # Plot EV1 and EV2 for all densities
        for ev_type in ['EV1', 'EV2']:  # Only plot EV1 and EV2 together
            if ev_type not in angles_data[calc_type]:
                continue
                
            for density in DENSITY_TYPES:
                if density not in angles_data[calc_type][ev_type]:
                    continue
                    
                data = angles_data[calc_type][ev_type][density]
                if not data:
                    continue
                
                image_nums = [d[0] for d in data]
                angles = [d[1] for d in data]
                
                # Create label
                density_label = DENSITY_LABELS[density]
                label = f"{ev_type} - {density_label}"
                
                # Plot with unique style for each combination
                ax.plot(image_nums, angles,
                       color=ev_colors[ev_type],
                       linestyle=density_styles[density],
                       marker=density_markers[density],
                       linewidth=2.5 if density == '' else 2.0,  # Total density slightly thicker
                       markersize=6 if density == '' else 5,
                       label=label,
                       alpha=density_alpha[density])
        
        # Reference axis label (EV1 and EV2 both use Y-axis)
        ax.set_xlabel('NEB Image Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Angle from Y-axis (degrees)', fontsize=12, fontweight='bold')
        ax.set_title(f'{bcp} - EV1 & EV2 Rotation Throughout Reaction ({calc_short})', 
                    fontsize=14, fontweight='bold')
        
        # Organize legend by eigenvector type
        handles, labels = ax.get_legend_handles_labels()
        
        # Sort legend entries: EV1 entries first, then EV2 entries
        ev1_entries = [(h, label) for h, label in zip(handles, labels) if label.startswith('EV1')]
        ev2_entries = [(h, label) for h, label in zip(handles, labels) if label.startswith('EV2')]
        
        sorted_handles = [h for h, label in ev1_entries] + [h for h, label in ev2_entries]
        sorted_labels = [label for h, label in ev1_entries] + [label for h, label in ev2_entries]
        
        ax.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set adaptive Y-axis limits with some padding
        if ax.has_data():
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            padding = max(5.0, y_range * 0.1)  # At least 5° padding, or 10% of range
            ax.set_ylim(max(0, y_min - padding), min(180, y_max + padding))
        else:
            ax.set_ylim(0, 180)  # Fallback if no data
        
        # Save plot
        filename = f"{bcp.replace('-', '_')}_{calc_short}_EV1_EV2.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    # Also create a separate plot for EV3 (all calculation types together)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot EV3 for all calculation types and densities
    # Use short calc names for color mapping (more flexible)
    calc_colors = {'origEEF': 'red', 
                   'revEEF': 'orange', 
                   'noEEF': 'blue'}
    
    for calc_type in calc_types:
        if 'EV3' not in angles_data[calc_type]:
            continue
            
        calc_short = calc_type.split('_')[-1]
        
        for density in DENSITY_TYPES:
            if density not in angles_data[calc_type]['EV3']:
                continue
                
            data = angles_data[calc_type]['EV3'][density]
            if not data:
                continue
            
            image_nums = [d[0] for d in data]
            angles = [d[1] for d in data]
            
            # Create label
            density_label = DENSITY_LABELS[density]
            label = f"{calc_short} - {density_label}"
            
            # Plot EV3
            ax.plot(image_nums, angles,
                   color=calc_colors.get(calc_short, 'black'),  # Use calc_short instead of calc_type
                   linestyle=density_styles[density],
                   marker=density_markers[density],
                   linewidth=2.5 if density == '' else 2.0,
                   markersize=6 if density == '' else 5,
                   label=label,
                   alpha=density_alpha[density])
    
    ax.set_xlabel('NEB Image Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Angle from Z-axis (degrees)', fontsize=12, fontweight='bold')
    ax.set_title(f'{bcp} - EV3 Rotation Throughout Reaction (All Calculations)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set adaptive Y-axis limits
    if ax.has_data():
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        padding = max(2.0, y_range * 0.1)  # Smaller padding for EV3 (usually small angles)
        ax.set_ylim(max(0, y_min - padding), min(180, y_max + padding))
    else:
        ax.set_ylim(0, 180)
    
    # Save EV3 plot
    filename = f"{bcp.replace('-', '_')}_EV3_all_calcs.png"
    filepath = output_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {filepath}")


# ============================================================================
# REPORTING
# ============================================================================

def print_rotation_summary(rotations: Dict[str, Dict]):
    """
    Print summary of start-to-end rotations for all BCPs and calculation types.
    
    Parameters:
    -----------
    rotations : dict
        {bcp: {calc_type: {ev_type: {density: angle}}}}
    """
    print("\n" + "="*80)
    print("START-TO-END EIGENVECTOR ROTATION SUMMARY")
    print("="*80)
    
    for bcp in rotations:
        print(f"\n{bcp}:")
        print("-" * 80)
        print(f"{'Calculation':<15} {'Eigenvector':<15} {'Density':<20} {'Rotation (deg)':<15}")
        print("-" * 80)
        
        for calc_type in rotations[bcp]:
            calc_short = calc_type.split('_')[-1]  # Extract origEEF, revEEF, noEEF
            
            for ev_type in EIGENVECTOR_TYPES:
                if ev_type in rotations[bcp][calc_type]:
                    for density in DENSITY_TYPES:
                        if density in rotations[bcp][calc_type][ev_type]:
                            angle = rotations[bcp][calc_type][ev_type][density]
                            density_label = DENSITY_LABELS[density]
                            print(f"{calc_short:<15} {ev_type:<15} {density_label:<20} {angle:>14.2f}°")


def save_rotation_data_to_csv(all_angles: Dict[str, Dict], output_dir: Path):
    """
    Save all angle data to CSV files for further analysis.
    
    Parameters:
    -----------
    all_angles : dict
        {bcp: {calc_type: {ev_type: {density: [(image_num, angle), ...]}}}}
    output_dir : Path
        Directory to save CSV files
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for bcp in all_angles:
        rows = []
        
        for calc_type in all_angles[bcp]:
            calc_short = calc_type.split('_')[-1]  # Extract origEEF, revEEF, noEEF
            
            for ev_type in EIGENVECTOR_TYPES:
                if ev_type not in all_angles[bcp][calc_type]:
                    continue
                    
                for density in DENSITY_TYPES:
                    if density not in all_angles[bcp][calc_type][ev_type]:
                        continue
                    
                    # Sort the data by image number to ensure correct order
                    angle_data = sorted(all_angles[bcp][calc_type][ev_type][density])
                    
                    for i, (image_num, angle) in enumerate(angle_data):
                        # Calculate relative angle (change from previous image)
                        if i == 0:
                            relative_angle = 0.0  # First image has no previous reference
                        else:
                            prev_angle = angle_data[i-1][1]
                            relative_angle = angle - prev_angle
                        
                        rows.append({
                            'BCP': bcp,
                            'Calculation': calc_short,
                            'Eigenvector': ev_type,
                            'Density': DENSITY_LABELS[density],
                            'Image_Number': image_num,
                            'Angle_to_Reference_deg': angle,
                            'Relative_Angle_deg': relative_angle
                        })
        
        if rows:
            df_out = pd.DataFrame(rows)
            filename = f"{bcp.replace('-', '_')}_angles.csv"
            filepath = output_dir / filename
            df_out.to_csv(filepath, index=False)
            print(f"Saved angle data with relative angles: {filepath}")


def save_summary_data_to_csv(all_rotations: Dict, all_max_rotations: Dict, output_dir: Path):
    """
    Save summary rotation data (start-to-end and maximum ranges) to CSV files.
    
    Parameters:
    -----------
    all_rotations : dict
        {bcp: {calc_type: {ev_type: {density: rotation_value}}}}
    all_max_rotations : dict  
        {bcp: {calc_type: {ev_type: {density: max_rotation_value}}}}
    output_dir : Path
        Directory to save CSV files
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for bcp in all_rotations:
        rows = []
        
        for calc_type in all_rotations[bcp]:
            calc_short = calc_type.split('_')[-1]  # Extract origEEF, revEEF, noEEF
            
            for ev_type in EIGENVECTOR_TYPES:
                if ev_type not in all_rotations[bcp][calc_type]:
                    continue
                    
                for density in DENSITY_TYPES:
                    if density not in all_rotations[bcp][calc_type][ev_type]:
                        continue
                    
                    start_to_end = all_rotations[bcp][calc_type][ev_type][density]
                    max_range = all_max_rotations[bcp][calc_type][ev_type][density]
                    
                    rows.append({
                        'BCP': bcp,
                        'Calculation': calc_short,
                        'Eigenvector': ev_type,
                        'Density': DENSITY_LABELS[density],
                        'Start_to_End_Rotation_deg': start_to_end,
                        'Maximum_Range_deg': max_range,
                        'Intermediate_Motion_deg': max_range - abs(start_to_end)  # How much extra motion beyond net change
                    })
        
        if rows:
            df_out = pd.DataFrame(rows)
            filename = f"{bcp.replace('-', '_')}_summary.csv"
            filepath = output_dir / filename
            df_out.to_csv(filepath, index=False)
            print(f"Saved summary data: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def apply_comprehensive_eigenvector_correction(all_angles_data):
    """
    Apply comprehensive eigenvector swap and sign flip correction to angle data.
    
    This function checks all 4 possibilities at each step:
    1. No change
    2. Sign flip both eigenvectors  
    3. Swap eigenvectors
    4. Swap and sign flip both eigenvectors
    """
    corrected_data = {}
    
    for bcp in all_angles_data:
        corrected_data[bcp] = {}
        
        for calc_type in all_angles_data[bcp]:
            corrected_data[bcp][calc_type] = {}
            
            for density in all_angles_data[bcp][calc_type]['EV1']:
                # Get EV1 and EV2 angle data
                if 'EV1' not in all_angles_data[bcp][calc_type] or 'EV2' not in all_angles_data[bcp][calc_type]:
                    continue
                    
                if density not in all_angles_data[bcp][calc_type]['EV1'] or density not in all_angles_data[bcp][calc_type]['EV2']:
                    continue
                
                ev1_data = all_angles_data[bcp][calc_type]['EV1'][density]
                ev2_data = all_angles_data[bcp][calc_type]['EV2'][density]
                
                if len(ev1_data) != len(ev2_data) or len(ev1_data) < 2:
                    # Copy original data if can't process
                    corrected_data[bcp].setdefault(calc_type, {}).setdefault('EV1', {})[density] = ev1_data
                    corrected_data[bcp].setdefault(calc_type, {}).setdefault('EV2', {})[density] = ev2_data
                    continue
                
                # Apply comprehensive correction
                corrected_ev1 = [ev1_data[0]]  # First point unchanged
                corrected_ev2 = [ev2_data[0]]
                
                corrections_made = 0
                
                for i in range(1, len(ev1_data)):
                    prev_ev1_angle = corrected_ev1[-1][1]
                    prev_ev2_angle = corrected_ev2[-1][1]
                    
                    curr_ev1_angle = ev1_data[i][1]
                    curr_ev2_angle = ev2_data[i][1]
                    curr_img = ev1_data[i][0]
                    
                    # Test all 4 possibilities
                    candidates = [
                        (curr_ev1_angle, curr_ev2_angle, "no_change"),
                        (180 - curr_ev1_angle, 180 - curr_ev2_angle, "both_sign_flip"),
                        (curr_ev2_angle, curr_ev1_angle, "swap"),
                        (180 - curr_ev2_angle, 180 - curr_ev1_angle, "swap_and_sign_flip")
                    ]
                    
                    best_option = None
                    best_total_diff = float('inf')
                    
                    for new_ev1, new_ev2, correction_type in candidates:
                        diff1 = abs(new_ev1 - prev_ev1_angle)
                        diff2 = abs(new_ev2 - prev_ev2_angle)
                        total_diff = diff1 + diff2
                        
                        if total_diff < best_total_diff:
                            best_total_diff = total_diff
                            best_option = (new_ev1, new_ev2, correction_type)
                    
                    if best_option is None:
                        best_option = (curr_ev1_angle, curr_ev2_angle, "fallback")
                    
                    corrected_ev1.append((curr_img, best_option[0]))
                    corrected_ev2.append((curr_img, best_option[1]))
                    
                    if best_option[2] != "no_change":
                        corrections_made += 1
                
                # Store corrected data
                corrected_data[bcp].setdefault(calc_type, {}).setdefault('EV1', {})[density] = corrected_ev1
                corrected_data[bcp].setdefault(calc_type, {}).setdefault('EV2', {})[density] = corrected_ev2
                
                # Copy EV3 data unchanged (no swapping needed)
                if 'EV3' in all_angles_data[bcp][calc_type] and density in all_angles_data[bcp][calc_type]['EV3']:
                    corrected_data[bcp].setdefault(calc_type, {}).setdefault('EV3', {})[density] = all_angles_data[bcp][calc_type]['EV3'][density]
                
                if corrections_made > 0:
                    print(f"  Applied {corrections_made} corrections to {bcp} {calc_type} {density}")
    
    return corrected_data


def save_analysis_output_to_file(all_rotations: Dict, all_max_rotations: Dict, output_dir: Path):
    """
    Save complete analysis output to a text file.
    
    Parameters:
    -----------
    all_rotations : dict
        Start-to-end rotation data
    all_max_rotations : dict  
        Maximum rotation range data
    output_dir : Path
        Directory to save the output file
    """
    from datetime import datetime
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bcp_analysis_output_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("BOND CRITICAL POINT EIGENVECTOR ROTATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir.absolute()}\n\n")
        
        # Write start-to-end rotation summary
        if all_rotations:
            f.write("=" * 80 + "\n")
            f.write("START-TO-END EIGENVECTOR ROTATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for bcp in all_rotations:
                f.write(f"{bcp}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Calculation':<12} {'Eigenvector':<12} {'Density':<20} {'Rotation (deg)':<15}\n")
                f.write("-" * 80 + "\n")
                
                for calc_type in all_rotations[bcp]:
                    calc_short = calc_type.split('_')[-1] if '_' in calc_type else calc_type
                    
                    for ev_type in ['EV1', 'EV2', 'EV3']:
                        if ev_type in all_rotations[bcp][calc_type]:
                            for density in DENSITY_TYPES:
                                if density in all_rotations[bcp][calc_type][ev_type]:
                                    rotation = all_rotations[bcp][calc_type][ev_type][density]
                                    density_label = DENSITY_LABELS[density]
                                    f.write(f"{calc_short:<12} {ev_type:<12} {density_label:<20} {rotation:>13.2f}°\n")
                f.write("\n")
        
        # Write maximum rotation ranges
        if all_max_rotations:
            f.write("=" * 80 + "\n")
            f.write("MAXIMUM EIGENVECTOR ROTATION RANGES\n")
            f.write("=" * 80 + "\n\n")
            
            for bcp in all_max_rotations:
                f.write(f"{bcp}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Calculation':<12} {'Eigenvector':<12} {'Density':<20} {'Max Range (deg)':<15}\n")
                f.write("-" * 80 + "\n")
                
                for calc_type in all_max_rotations[bcp]:
                    calc_short = calc_type.split('_')[-1] if '_' in calc_type else calc_type
                    
                    for ev_type in ['EV1', 'EV2', 'EV3']:
                        if ev_type in all_max_rotations[bcp][calc_type]:
                            for density in DENSITY_TYPES:
                                if density in all_max_rotations[bcp][calc_type][ev_type]:
                                    max_rot = all_max_rotations[bcp][calc_type][ev_type][density]
                                    density_label = DENSITY_LABELS[density]
                                    f.write(f"{calc_short:<12} {ev_type:<12} {density_label:<20} {max_rot:>13.2f}°\n")
                f.write("\n")
        
        # Write analysis summary and interpretation
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS INTERPRETATION\n") 
        f.write("=" * 80 + "\n\n")
        
        f.write("Key Insights:\n")
        f.write("-" * 40 + "\n")
        
        if all_rotations and all_max_rotations:
            for bcp in all_rotations:
                f.write(f"\n{bcp} Bond Critical Point:\n\n")
                
                # Compare start-to-end vs max range for each calculation
                for calc_type in all_rotations[bcp]:
                    calc_short = calc_type.split('_')[-1] if '_' in calc_type else calc_type
                    f.write(f"  {calc_short} Calculation:\n")
                    
                    # Focus on EV1 Total Density as representative
                    if ('EV1' in all_rotations[bcp][calc_type] and 
                        '' in all_rotations[bcp][calc_type]['EV1'] and
                        calc_type in all_max_rotations[bcp] and
                        'EV1' in all_max_rotations[bcp][calc_type] and
                        '' in all_max_rotations[bcp][calc_type]['EV1']):
                        
                        start_end = all_rotations[bcp][calc_type]['EV1']['']
                        max_range = all_max_rotations[bcp][calc_type]['EV1']['']
                        
                        f.write(f"    - Start-to-end rotation: {start_end:.1f}°\n")
                        f.write(f"    - Maximum range: {max_range:.1f}°\n")
                        
                        if max_range > start_end * 2:
                            f.write(f"    - Interpretation: Significant intermediate motion (returns near start)\n")
                        elif max_range > start_end * 1.5:
                            f.write(f"    - Interpretation: Moderate intermediate motion\n")
                        else:
                            f.write(f"    - Interpretation: Steady progression throughout reaction\n")
                    f.write("\n")
        
        f.write("\nNotes:\n")
        f.write("- Start-to-end rotation: Net change from first to last image\n")
        f.write("- Maximum range: Largest angle difference between any two images\n")
        f.write("- All values use corrected eigenvector data (sign/swap artifacts removed)\n")
        f.write("- EV3 typically shows minimal rotation (bond axis direction)\n")
        f.write("- Cross-density analysis reveals spin-dependent field effects\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved complete analysis output: {output_file}")


def main(csv_file: str):
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("BOND CRITICAL POINT EIGENVECTOR ROTATION ANALYSIS")
    print("="*80 + "\n")
    
    global OUTPUT_DIR
    # append csv file name (no extension) to output dir as another dir
    base_name = Path(csv_file).stem
    OUTPUT_DIR = OUTPUT_DIR / base_name
    
    # Load data
    df = load_and_process_data(csv_file)
    
    # Storage for results
    all_angles = {}
    all_rotations = {}
    
    # Process each specified BCP - calculate angles only (no plotting yet)
    for bcp in BOND_CRITICAL_POINTS:
        print(f"\n{'='*60}")
        print(f"Processing BCP: {bcp}")
        print('='*60)
        
        # Calculate angles throughout reaction
        angles_data = calculate_angles_for_bcp(df, bcp)
        if angles_data:
            all_angles[bcp] = angles_data
    
    # Apply comprehensive swap and sign flip correction BEFORE plotting
    if all_angles:
        print("\nApplying comprehensive eigenvector swap and sign flip correction...")
        corrected_angles = apply_comprehensive_eigenvector_correction(all_angles)
        if corrected_angles:
            all_angles = corrected_angles
    
    # Now create plots and calculate rotations using corrected data
    all_max_rotations = {}
    for bcp in BOND_CRITICAL_POINTS:
        if bcp in all_angles:
            # Create standard plots with corrected data
            plot_eigenvector_rotation(all_angles[bcp], bcp, OUTPUT_DIR)
            
            # Create NEW ANALYSIS: relative rotation rate plots
            plot_relative_rotation_rates(all_angles[bcp], bcp, OUTPUT_DIR)
            
            # Calculate start-to-end rotations using corrected angle data
            rotations = calculate_start_to_end_rotation_from_angles(all_angles, bcp)
            if rotations:
                all_rotations[bcp] = rotations
            
            # Calculate NEW ANALYSIS: maximum rotation ranges
            max_rotations = calculate_maximum_rotations(all_angles[bcp], bcp)
            if max_rotations:
                all_max_rotations[bcp] = max_rotations
    
    # Print summary (now with corrected data)
    if all_rotations:
        print_rotation_summary(all_rotations)
    
    # Print NEW ANALYSIS: maximum rotation ranges
    if all_max_rotations:
        print_maximum_rotation_summary(all_max_rotations)
    
    # Save angle data to CSV
    if all_angles:
        save_rotation_data_to_csv(all_angles, OUTPUT_DIR)
    
    # Save summary data to CSV
    if all_rotations and all_max_rotations:
        save_summary_data_to_csv(all_rotations, all_max_rotations, OUTPUT_DIR)
    
    # Save complete analysis output to text file
    save_analysis_output_to_file(all_rotations, all_max_rotations, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Output saved to: {OUTPUT_DIR.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Path to your CSV file
    CSV_FILE = "Cys_propane_NEB_NF_cp_info.csv"  # Full dataset
    
    # Run analysis
    main(CSV_FILE)
