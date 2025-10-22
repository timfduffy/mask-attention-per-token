# Grid Layout Analysis Summary

## File: eras_text_orb_looking_at.parquet

### Key Finding
The parquet file contains **4 different prompts** with **different grid layouts**!

### Prompts Breakdown

| Prompt | Total Positions | Vision Tokens | Text Tokens | Grid Size |
|--------|----------------|---------------|-------------|-----------|
| eras_text | 549 | 528 | 21 | 23x23 (approx) |
| eras_figures | 550 | 528 | 22 | 23x23 (approx) |
| eras_legs | 549 | 528 | 21 | 23x23 (approx) |
| **orb_looking_at** | **596** | **576** | **20** | **24x24 ✓** |

### Position 549 Confusion

You saw "position 549 as the highest" because you were likely looking at one of the "eras_*" prompts where:
- **eras_figures**: Position 549 is TEXT (`'\n'`) - the last position before vision tokens end
- The eras prompts only have **528 vision tokens**, not 576

For the **orb_looking_at** prompt:
- Position 549 IS a vision token (`'<|image_pad|>'`)
- This prompt has the full **576 vision tokens** (24x24 grid)

### Image Resolution Math

**orb_looking_at** (24x24 grid):
```
1024x1024 original → 768x768 downscaled
768 / 32 = 24 (using 32x32 pixel patches)
24 x 24 = 576 tokens ✓
```

**eras_* prompts** (≈23x23 grid):
```
528 tokens = approximately 23x23 grid
sqrt(528) ≈ 22.98
Closest factors: 22x24 = 528
```

### Visualization Behavior

The `visualize_vl_grid.html` tool:
1. **Filters by token content** (looks for 'image_pad', 'vision_token', etc.)
2. **Ignores position numbers** when determining what's a vision token
3. **Groups by prompt_name** so each prompt is visualized separately
4. **Auto-detects grid size** using `sqrt(num_tokens)` or `ceil(sqrt(num_tokens))`

For "eras_figures" with 528 tokens:
- sqrt(528) ≈ 22.98
- ceil(22.98) = 23
- Displays as: **23x23 grid** (with 1 empty cell)

For "orb_looking_at" with 576 tokens:
- sqrt(576) = 24
- Displays as: **24x24 grid** (perfect!)

### Recommendation

When viewing the data in the visualization tool:
1. **Select which prompt to view** using a dropdown/filter
2. Each prompt will show its correct grid dimensions
3. The "orb_looking_at" prompt will display as a perfect 24x24 grid

The test scripts have been saved in your directory for future reference:
- `analyze_grid_layout.py` - Basic grid dimension analysis
- `analyze_grid_gaps.py` - Check for gaps in vision token positions
- `check_pos_549.py` - Verify specific position contents
- `investigate_duplicate_positions.py` - Understand multi-prompt structure

