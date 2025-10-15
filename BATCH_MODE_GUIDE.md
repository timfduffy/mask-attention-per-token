# Batch Mode Guide

The masking analysis tool now supports batch processing of multiple prompts using YAML configuration files!

## Quick Start

### Batch Mode (Multiple Prompts)
```bash
python mask_impact_analysis.py --config prompts_config.yaml
```

### Single Prompt Mode (Legacy)
```bash
# From prompt.txt (default)
python mask_impact_analysis.py --num-tokens 5

# From specific file
python mask_impact_analysis.py --prompt my_prompt.txt --num-tokens 10

# Inline prompt
python mask_impact_analysis.py --prompt "The capital of France is" --num-tokens 3
```

## YAML Configuration Format

Create a `prompts_config.yaml` file:

```yaml
model:
  path: "path/to/your/model"
  
device: "auto"  # "auto", "cuda", or "cpu"

prompts:
  - name: "experiment1"
    enabled: true  # Optional: defaults to true if omitted
    prompt: "Your prompt text here"
    num_tokens: 5
    
  - name: "experiment2"
    enabled: false  # Set to false to skip this prompt
    prompt: "prompt.txt"  # Load from file
    num_tokens: 10
    
  - name: "experiment3"
    prompt: |
      Multi-line
      prompt
      goes here
    num_tokens: 3
```

## Output Files

All output files are automatically saved to the `output/` directory (created if it doesn't exist).

### Batch Mode
Each prompt generates its own files:
- `output/{name}_results.csv` - Spreadsheet-friendly format
- `output/{name}_results.json` - Fast loading for visualization

Example: `output/experiment1_results.csv`, `output/experiment1_results.json`

### Single Prompt Mode
Default files:
- `output/masking_results.csv`
- `output/masking_results.json`

Or custom name with `--output`:
```bash
python mask_impact_analysis.py --prompt "test" --output my_experiment
# Creates: output/my_experiment.csv, output/my_experiment.json
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to YAML config file | `--config prompts.yaml` |
| `--prompt` | Single prompt text or .txt file | `--prompt "The sky is"` |
| `--num-tokens` | Number of tokens to generate (single mode) | `--num-tokens 5` |
| `--output` | Output file basename (single mode) | `--output my_results` |
| `--model` | Override model path | `--model path/to/model` |
| `--device` | Force device selection | `--device cuda` |

## Examples

### Run 3 experiments in one go
```bash
python mask_impact_analysis.py --config prompts_config.yaml
```

### Quick test with inline prompt
```bash
python mask_impact_analysis.py --prompt "Hello world" --num-tokens 1 --output hello_test
```

### Override device for batch run
```bash
python mask_impact_analysis.py --config prompts_config.yaml --device cpu
```

### Use different model
```bash
python mask_impact_analysis.py --config prompts_config.yaml --model "Qwen/Qwen3-0.6B"
```

## Early Stopping

The script automatically stops generation when the model produces an end-of-sequence token (`<|im_end|>` or EOS). This:
- Saves computation time
- Makes analysis more realistic (no need to analyze padding tokens)
- Works even if you set `num_tokens: 100` - it'll stop when the model says it's done

Example output:
```
Generated sequence (5/20 tokens, stopped early at 5 tokens):
  The answer is 12.<|im_end|>

⚠ Generation stopped early: Model generated end token
```

## Tips

1. **Start small**: Test with `num_tokens: 1` first to check performance
2. **Organize experiments**: Use descriptive names like `code_completion`, `qa_medical`, etc.
3. **Reuse prompts**: Reference the same `.txt` file in multiple config entries with different `num_tokens`
4. **GPU acceleration**: Use `--device cuda` for faster processing
5. **Visualization**: Load the generated JSON files into `visualize_results.html` for interactive exploration
6. **Toggle experiments**: Use `enabled: false` to temporarily skip prompts without deleting them from the config
7. **Iterative testing**: Start by enabling only 1-2 prompts, verify results, then enable more
8. **Set generous `num_tokens`**: Don't worry about setting exact lengths - early stopping will handle it (e.g., use `num_tokens: 50` for short answers)

