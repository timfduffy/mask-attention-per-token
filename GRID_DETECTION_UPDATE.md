# Grid Detection Update - Smart Rectangular Grid Support

## Problem
The visualization tool was forcing all grids to be square by using `Math.ceil(sqrt(tokens))`, which caused:
- **eras_text** (528 tokens) to display as **23×23 grid** with 1 empty cell
- Should have been **24×22** (exact fit, no empty cells)

## Solution Implemented
Added smart rectangular grid detection that:
1. Detects perfect squares (like 576 → 24×24)
2. For non-squares, finds all factor pairs
3. Selects the pair closest to square (minimal width-height difference)

## Changes Made

### `visualize_vl_grid.html`

**Added `findBestGrid()` function:**
```javascript
function findBestGrid(numTokens) {
    const sqrtExact = Math.sqrt(numTokens);
    
    // Perfect square? Use it
    if (Number.isInteger(sqrtExact)) {
        return { width: sqrtExact, height: sqrtExact };
    }
    
    // Find all factor pairs
    const factors = [];
    for (let i = 1; i <= Math.sqrt(numTokens) + 1; i++) {
        if (numTokens % i === 0) {
            const j = Math.floor(numTokens / i);
            factors.push({ width: j, height: i });
        }
    }
    
    // Prefer factors closest to square
    const best = factors.reduce((prev, curr) => {
        const prevDiff = Math.abs(prev.width - prev.height);
        const currDiff = Math.abs(curr.width - curr.height);
        return currDiff < prevDiff ? curr : prev;
    });
    
    return best;
}
```

**Updated rendering:**
- Changed from `gridSize` to `gridDims` object with `width` and `height`
- Display shows actual dimensions: "24×22" instead of "23×23"
- Added "Cells" stat showing total grid cells
- CSS Grid uses proper width for columns

## Verification Results

All files now display with correct dimensions:

| File | Vision Tokens | Old Display | New Display | Status |
|------|---------------|-------------|-------------|--------|
| eras_text | 528 | 23×23 (529) | **24×22 (528)** | ✓ Fixed |
| eras_figures | 528 | 23×23 (529) | **24×22 (528)** | ✓ Fixed |
| eras_legs | 528 | 23×23 (529) | **24×22 (528)** | ✓ Fixed |
| orb_looking_at | 576 | 24×24 (576) | **24×24 (576)** | ✓ Same |

## How It Works

For **528 tokens** (eras_* files):
```
All possible grids: 528×1, 264×2, 176×3, 132×4, 88×6, 66×8, 
                    48×11, 44×12, 33×16, 24×22, 22×24

Closest to square: 24×22 (difference: 2)
Second closest: 22×24 (difference: 2)
Algorithm picks: 24×22 (wider dimension first)
```

For **576 tokens** (orb_looking_at):
```
Perfect square: 24×24
Algorithm picks: 24×24
```

## Benefits

1. **No empty cells** - Every cell in the grid corresponds to an actual token
2. **Correct aspect ratio** - Matches the actual image processing grid
3. **Better visualization** - Overlay image aligns properly with tokens
4. **Automatic** - No manual configuration needed
5. **Robust** - Works for any token count with factors

## Image Processing Context

**768×768 image with 32×32 patches:**
- Perfect fit: 24×24 = 576 tokens ✓

**768×704 image with 32×32 patches:**  
- Width: 768 / 32 = 24
- Height: 704 / 32 = 22
- Total: 24×22 = 528 tokens ✓

This suggests the eras_* images were resized to 768×704 to preserve aspect ratio from the original resolution.

## Testing

Created test files:
- `test_grid_detection.html` - Interactive test showing all cases
- `verify_grid_dimensions.py` - Verification script for parquet files

Run verification:
```bash
python verify_grid_dimensions.py
```

## Transparent Grid Overlay

The rectangular grid support works seamlessly with the transparent overlay feature:
- Grid cells remain square (1:1 aspect ratio)
- Overall grid is rectangular (24:22 ratio)
- Background image scales to fit the rectangular grid container
- Heatmap colors blend properly with transparency

