# Breakpoint-Based Group Detection Analysis

This document describes the new breakpoint-based approach for analyzing group detection results, which replaces the traditional rolling window method.

## Overview

Instead of using fixed-size rolling windows, this approach identifies **breakpoints** where group detection results change across any of the four clues (head, shoulder, hip, foot) and uses these breakpoints as natural window boundaries.

## Key Differences from Rolling Windows

### Traditional Rolling Windows
- Fixed window size (e.g., 60 frames)
- Fixed step size (e.g., 60 frames)
- Windows may contain multiple different group structures
- Requires aggregation methods to handle multiple detections per window
- Processes all cameras together

### Breakpoint-Based Windows
- Variable window size based on when group structures change
- Each window contains only one consistent group structure
- Uses `closest_to_start` aggregation (since all detections within a window are identical)
- More semantically meaningful windows
- **Processes each camera individually** for more granular analysis

## Example

Consider the following scenario:

**Head clue detection:**
- Seconds 1-10: `{[1,2], [3,4]}`
- Second 11: Changes to `{[1,2,3], [4]}`

**Shoulder clue detection:**
- Seconds 1-15: `{[1], [2,3,4]}`
- Second 16: Changes to something else

**Breakpoint-based windows:**
- Window 1: Seconds 1-10 (head groups change at 11)
- Window 2: Seconds 11-15 (shoulder groups change at 16)
- Window 3: Seconds 16+ (until next breakpoint)

## Files

### Main Script
- `detectGroupNumBreakpoints.m` - Main analysis script using breakpoint-based windows

### Test Script
- `testBreakpointDetection.m` - Demonstrates the breakpoint detection process

## Usage

### Running the Main Analysis

```matlab
% Run the breakpoint-based analysis
run detectGroupNumBreakpoints.m
```

This will:
1. Load the required data (`data_results.mat`, `speaking_status.mat`)
2. Find breakpoints for each camera individually across all four clues
3. Generate windows based on these breakpoints for each camera
4. Run `countSpeakerGroups` with `closest_to_start` aggregation
5. Process results with `processWindowTable`

### Testing the Approach

```matlab
% Test breakpoint detection on specific video/camera
run testBreakpointDetection.m
```

This will show:
- Sample group detections over time
- Identified breakpoints
- Generated windows
- Comparison with rolling window approach

## Key Functions

### `findBreakpointsForCamera(data_results, clues, vid, cam)`
- Finds all timestamps where group detection results change for a specific camera
- Considers all specified clues for the given camera
- Returns sorted array of breakpoint timestamps

### `generateBreakpointWindows(breakpoints, vid, cam)`
- Creates windows between consecutive breakpoints for a specific camera
- Returns cell array of window structures with start/end times

### `getSpeakingStatusForWindow(actions, window_start, window_end)`
- Extracts speaking status for a specific window
- Identifies participants who spoke continuously throughout the window

## Advantages

1. **Semantic Meaning**: Each window represents a period of consistent group structure
2. **No Aggregation Needed**: Since all detections within a window are identical, `closest_to_start` is sufficient
3. **Adaptive Window Size**: Window sizes adapt to the natural rhythm of group changes
4. **Reduced Noise**: Avoids mixing different group structures within the same window
5. **Camera-Specific Analysis**: Each camera is processed individually, providing more granular insights
6. **Better Comparison**: Enables direct comparison of group dynamics across different camera viewpoints

## Output

The script produces the same output format as the original `detectGroupNum.m`:
- `window_table`: Table with window information and group detection results
- `filtered_table`: Filtered results after processing
- `pairwise_diffs`: Pairwise differences between detection methods

## Integration

This approach integrates seamlessly with existing analysis functions:
- `countSpeakerGroups()` - Works with breakpoint-based windows
- `processWindowTable()` - Processes results in the same format
- All visualization and analysis tools remain compatible

## Performance

The breakpoint-based approach typically creates fewer windows than rolling windows, leading to:
- Faster processing
- More meaningful statistical analysis
- Reduced computational overhead
