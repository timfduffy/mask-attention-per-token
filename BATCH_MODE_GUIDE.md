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
    prompt: "Your prompt text here"
    num_tokens: 5
    
  - name: "experiment2"
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

### Batch Mode
Each prompt generates its own files:
- `{name}_results.csv` - Spreadsheet-friendly format
- `{name}_results.json` - Fast loading for visualization

Example: `experiment1_results.csv`, `experiment1_results.json`

### Single Prompt Mode
Default files:
- `masking_results.csv`
- `masking_results.json`

Or custom name with `--output`:
```bash
python mask_impact_analysis.py --prompt "test" --output my_experiment
# Creates: my_experiment.csv, my_experiment.json
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

## Tips

1. **Start small**: Test with `num_tokens: 1` first to check performance
2. **Organize experiments**: Use descriptive names like `code_completion`, `qa_medical`, etc.
3. **Reuse prompts**: Reference the same `.txt` file in multiple config entries with different `num_tokens`
4. **GPU acceleration**: Use `--device cuda` for faster processing
5. **Visualization**: Load the generated JSON files into `visualize_results.html` for interactive exploration

