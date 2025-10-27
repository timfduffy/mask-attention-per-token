# Vision-Language Grid Viewer Guide

## Overview

The VL Grid Viewer (`visualize_vl_grid.html`) is a specialized visualization tool designed for analyzing **image token masking effects** in Vision-Language models. Unlike the standard table viewer, this displays image tokens in their natural **2D grid structure**, making spatial patterns immediately visible.

## Key Features

### 🖼️ 2D Image Token Grid
- Image tokens displayed as a square grid (auto-detects dimensions)
- Each cell shows the masking impact value
- Spatial patterns visible at a glance
- Grid dimensions shown in stats (e.g., "16×16" for 256 tokens)

### 📝 Separate Text Token Display
- Text tokens shown horizontally below the image grid
- Clear separation between modalities
- Token text displayed for readability
- Position numbers for reference

### 🎨 Dual Heatmap Coloring
- **Image tokens**: Colored relative to max image token value
- **Text tokens**: Colored relative to max text token value
- Prevents image tokens from dominating the color scale
- Shows relative importance within each modality

### ⌨️ Keyboard Navigation

**Layer Navigation:**
- `←` (Left Arrow): Previous layer
- `→` (Right Arrow): Next layer

**Generation Step Navigation:**
- `↑` (Up Arrow): Previous generation step
- `↓` (Down Arrow): Next generation step

**Quick Toggles:**
- `M`: Toggle between L2 Distance and Cosine Distance
- `S`: Toggle between Linear and Square Root scaling

### 📊 Real-time Statistics
- **Per-modality stats**: Max, average values for image and text separately
- **Grid dimensions**: Shows detected image grid size
- **Token counts**: Number of image vs text tokens
- **Current view**: Layer, step, metric, variant, scaling all visible

### 🎯 Image Overlay
- **Load the original image** to display as a background under the grid
- **Grid patches align** with how the image was tokenized
- **Transparent Grid Overlay** checkbox to make patches semi-transparent
- **Spatial correspondence** between image regions and token importance
- Makes it easy to see which image regions have high masking impact

## Usage

### Basic Workflow

1. **Open the viewer**
   ```bash
   # In browser, open:
   visualize_vl_grid.html
   ```

2. **Load your data**
   - Click "Load Parquet" button
   - Select your `.parquet` file
   - Viewer starts at Layer 0, Step 0

3. **Load image (optional)**
   - Click "Load Image" button
   - Select the original image file
   - Image will display as a background under the grid
   - Grid patches align with how the image was tokenized
   - Check "Transparent Grid Overlay" to see the image more clearly

4. **Navigate**
   - Use arrow keys to explore layers and steps
   - Watch how patterns evolve across layers
   - See how generation progresses across steps

5. **Analyze**
   - Compare image vs text token importance
   - Identify spatial patterns in image attention
   - Toggle metrics to see different perspectives

### Example Analysis Session

```
1. Load: output/image_question.parquet
   → Shows Layer 0, Step 0
   → Image grid: 16×16 (256 tokens)
   → Text tokens: 27 tokens

2. Press → (5 times)
   → Now at Layer 5
   → Notice how image token patterns change

3. Press ↓
   → Now at Step 1 (after generating first token)
   → See how attention shifts

4. Press M
   → Switch to Cosine Distance
   → Different perspective on same data

5. Press S
   → Square root scaling
   → Emphasizes lower values
```

## Understanding the Display

### Image Grid Section

```
🖼️ Image Tokens (256)
┌─────┬─────┬─────┬─────┐
│0.123│0.145│0.098│0.156│  ← Each cell shows:
│Pos 0│Pos 1│Pos 2│Pos 3│     - Value (3 decimals)
├─────┼─────┼─────┼─────┤     - Position number
│0.178│0.134│0.167│0.145│
│Pos 4│Pos 5│Pos 6│Pos 7│
└─────┴─────┴─────┴─────┘

Stats: Max: 0.1780  Avg: 0.1420  Grid: 16×16
```

### Text Tokens Section

```
📝 Text Tokens (27)
┌────────┐┌────────┐┌────────┐
│  What  ││   is   ││  this  │  ← Each token shows:
│ 0.089  ││ 0.045  ││ 0.234  │     - Token text
└────────┘└────────┘└────────┘     - Value (3 decimals)

Stats: Max: 0.2340  Avg: 0.1120
```

### Color Interpretation

**White → Red gradient:**
- White: Lowest impact (relative to max in that modality)
- Light pink: Low impact
- Pink: Medium impact
- Red: High impact
- Dark red (white text): Maximum impact

**Important:** Image and text use **separate scales**!

## Advanced Features

### Scaling Modes

**Linear (default):**
- Direct proportional coloring
- `color_intensity = value / max_value`
- Good for absolute comparisons

**Square Root:**
- Emphasizes lower values more
- `color_intensity = sqrt(value / max_value)`
- Good for finding subtle patterns

**Percentile:**
- Based on rank among all values
- `color_intensity = percentile_rank / 100`
- Good for seeing relative distribution
- Less affected by outliers

### Exclude First Image Token

Just like text tokens can have an "attention sink" at position 0, image tokens may have a similar pattern. Check "Exclude 1st Image Token" to:
- Exclude position 0 from max value calculation
- Prevents attention sink from dominating color scale
- Better reveals patterns in remaining tokens
- Enabled by default

### Transparent Grid Overlay

When you load an image, check "Transparent Grid Overlay" to:
- Make grid cells semi-transparent
- See the underlying image more clearly
- Understand which image regions correspond to which tokens
- Visualize spatial attention patterns over the actual image
- Disabled by default

### Square Grid Patches

Image tokens are rendered as perfect square patches that align edge-to-edge:
- No gaps between cells (`gap: 0`)
- Square aspect ratio enforced (`aspect-ratio: 1/1`)
- Minimum 50×50px size for visibility
- Border around entire grid for clear boundaries
- Designed for overlay on actual images in future versions
- Each patch corresponds to one image region/token

### Detecting Image Grid Dimensions

The viewer automatically detects grid size:
- Excludes special vision tokens (`<vision_start>`, `<vision_end>`, `<image>`)
- Takes square root of remaining image token count
- Uses exact square root for perfect squares (256 → 16, not 17)
- Rounds up for non-perfect squares
- Example: 256 `<image_pad>` tokens → 16×16 grid
- Example: 1024 tokens → 32×32 grid

### Token Identification

**Image tokens** detected by (case-insensitive):
- Contains "vision_token"
- Contains "image_pad"
- Contains "<image>"
- Contains "vision_start"
- Starts with "img_"

**Text tokens**:
- Everything else (any token not matching above patterns)

## Tips & Tricks

### Finding Important Tokens

1. **Start at Layer 0** to see initial attention patterns
2. **Navigate forward** (`→`) through layers to see evolution
3. **Look for clusters** in the image grid - often semantically meaningful
4. **Compare to text tokens** - which modality dominates?

### Analyzing Generation

1. **Start at Step 0** (analyzing prompt)
2. **Press ↓** to see Step 1 (after 1st generated token)
3. **Continue ↓** to see how attention evolves as generation proceeds
4. **Look for shifts** - does attention move from image to text?

### Performance Optimization

**Fast Navigation:**
- Keyboard is much faster than mouse
- Hold arrow keys for rapid scanning
- Use `M` and `S` for quick comparisons

**Large Files:**
- Viewer loads all data once (fast with Parquet)
- Rendering is instant (only shows one layer/step)
- No performance issues even with millions of rows

## Troubleshooting

### Grid looks wrong
- **Issue**: Image tokens not detected correctly
- **Fix**: Check token naming in your data
- **Supported names**: 
  - "vision_token_X", "img_X" (numeric suffixes)
  - "<image_pad>", "<image>", "<vision_start>" (special tokens)
  - Any token containing these strings (case-insensitive)

### No data shown
- **Issue**: No tokens match current layer/step/variant
- **Check**: Try different variant in dropdown
- **Check**: Console for error messages

### Keyboard navigation not working
- **Issue**: Browser focus not on page
- **Fix**: Click anywhere on page first
- **Note**: Some keys (like arrow keys) may scroll page

### Colors look washed out
- **Try**: Toggle to Square Root scaling (`S` key)
- **Try**: Check if max values are very small
- **Check**: Statistics section for actual values

## Comparison with Standard Viewer

| Feature | Standard Viewer | VL Grid Viewer |
|---------|----------------|----------------|
| **Layout** | Table (all layers/steps) | Grid (one layer/step) |
| **Image tokens** | Linear list | 2D grid |
| **Text tokens** | Same as image | Separate section |
| **Color scale** | Single global | Dual (image/text) |
| **Navigation** | Mouse + dropdowns | Keyboard + dropdowns |
| **Best for** | Overview, filtering | Spatial analysis |
| **Performance** | Filter by step needed | Always fast |

## Use Cases

### 1. Spatial Pattern Analysis
**Goal**: Find which image regions are most important

**Method**:
1. Load file, go to Layer 10+
2. Look at image grid heatmap
3. Identify hot spots (red cells)
4. Note their spatial arrangement

**Insight**: Important regions often cluster (e.g., objects, faces)

### 2. Layer Evolution
**Goal**: See how patterns change across layers

**Method**:
1. Start at Layer 0
2. Press → repeatedly to scan through layers
3. Watch how colors shift
4. Note which layers show most variation

**Insight**: Early layers = low-level, late layers = semantic

### 3. Generation Analysis
**Goal**: Understand how attention shifts during generation

**Method**:
1. Go to Layer 20+ (high-level reasoning)
2. Start at Step 0
3. Press ↓ to advance through steps
4. Watch image vs text balance

**Insight**: Often shifts from image-heavy to text-heavy

### 4. Modality Comparison
**Goal**: Which modality dominates?

**Method**:
1. Load any layer/step
2. Compare max values in stats
3. Compare average values
4. Look at heatmap distributions

**Insight**: Text often has higher individual tokens, image has broader spread

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| `←` | Previous layer |
| `→` | Next layer |
| `↑` | Previous generation step |
| `↓` | Next generation step |
| `M` | Toggle metric (L2 ↔ Cosine) |
| `S` | Cycle scaling (Linear → Sqrt → Percentile) |

## Technical Notes

### Data Format
- Expects standard Parquet format from `mask_impact_vl.py`
- Required columns: `layer`, `generation_step`, `token_masked`, `token_position`, `variant`, `l2_distance`, `cosine_distance`
- Image tokens detected by: "vision_token", "image_pad", "<image>", "vision_start", or "img_" prefix (case-insensitive)

### Performance
- Loads entire Parquet file at once (typically <1 second)
- Renders only current layer/step (instant)
- Smooth navigation even with 3M+ row datasets
- No memory issues (data compressed in memory)

### Browser Compatibility
- Requires ES6 module support (all modern browsers)
- Tested on Chrome, Firefox, Edge, Safari
- Uses hyparquet for Parquet parsing
- No build step or dependencies needed

## Attention Weight Visualization

When enabled in the config with `save_attention_weights: true`, you can view raw attention weights instead of masking impact distances.

### Attention Weight Variants

**Per-Head Attention Weights (`AttentionWeight_Head_0`, `AttentionWeight_Head_1`, ...):**
- Shows how much each individual attention head attends to each token
- Values are probabilities (0.0 to 1.0) after softmax
- Allows you to see which heads focus on which tokens
- Each head can have different attention patterns

**Averaged Attention Weight (`AttentionWeight_Avg`):**
- Average attention weight across all heads
- Gives an overall picture of token importance
- Useful for identifying globally important tokens

### Key Differences from Distance Metrics

When viewing attention weight variants:
- **Metric selector is disabled** - attention weights are not computed per distance metric
- **Values are probabilities** (0.0 to 1.0), not distances
- **Higher values = more attention**, unlike distances where lower can mean more similar
- **Direct measurement** of attention, not impact of masking

### Use Cases

- **Visualize attention patterns**: See which image regions the model focuses on
- **Per-head analysis**: Identify specialized heads (e.g., one head for edges, another for objects)
- **Compare with masking impact**: Does high attention correlate with high masking impact?
- **Debug model behavior**: Understand why the model made certain predictions

## Future Enhancements

Possible improvements:
- Click token to see details/history
- Compare two layers side-by-side
- Animate through layers/steps
- Export current view as image
- 3D view (layers as depth dimension)

---

**Created for the mask_impact project**  
**Analyzing token-level attention in Vision-Language models**

