# Qwen3-VL

[Qwen3-VL](https://huggingface.co/papers/2502.13923) is a multimodal vision-language model series, encompassing both dense and MoE variants, as well as Instruct and Thinking versions. Building upon its predecessors, Qwen3-VL delivers significant improvements in visual understanding while maintaining strong pure text capabilities. Key architectural advancements include: enhanced MRope with interleaved layout for better spatial-temporal modeling, DeepStack integration to effectively leverage multi-level features from the Vision Transformer (ViT), and improved video understanding through text-based time alignment—evolving from T-RoPE to text timestamp alignment for more precise temporal grounding. These innovations collectively enable Qwen3-VL to achieve superior performance in complex multimodal tasks.

Model usage

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL")
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }

]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
</hfoptions>

## Qwen3VLConfig[[transformers.Qwen3VLConfig]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLConfig</name><anchor>transformers.Qwen3VLConfig</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py#L215</source><parameters>[{"name": "text_config", "val": " = None"}, {"name": "vision_config", "val": " = None"}, {"name": "image_token_id", "val": " = 151655"}, {"name": "video_token_id", "val": " = 151656"}, {"name": "vision_start_token_id", "val": " = 151652"}, {"name": "vision_end_token_id", "val": " = 151653"}, {"name": "tie_word_embeddings", "val": " = False"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **text_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLTextConfig`) --
  The config object or dictionary of the text backbone.
- **vision_config** (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLVisionConfig`) --
  The config object or dictionary of the vision backbone.
- **image_token_id** (`int`, *optional*, defaults to 151655) --
  The image token index to encode the image prompt.
- **video_token_id** (`int`, *optional*, defaults to 151656) --
  The video token index to encode the image prompt.
- **vision_start_token_id** (`int`, *optional*, defaults to 151652) --
  The start token index to encode the image prompt.
- **vision_end_token_id** (`int`, *optional*, defaults to 151653) --
  The end token index to encode the image prompt.
- **tie_word_embeddings** (`bool`, *optional*, defaults to `False`) --
  Whether to tie the word embeddings.</paramsdesc><paramgroups>0</paramgroups></docstring>

This is the configuration class to store the configuration of a [Qwen3VLModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLModel). It is used to instantiate a
Qwen3-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen3-VL-4B-Instruct [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig) for more information.




<ExampleCodeBlock anchor="transformers.Qwen3VLConfig.example">

```python
>>> from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

>>> # Initializing a Qwen3-VL style configuration
>>> configuration = Qwen3VLConfig()

>>> # Initializing a model from the Qwen3-VL-4B style configuration
>>> model = Qwen3VLForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

</ExampleCodeBlock>

</div>

## Qwen3VLTextConfig[[transformers.Qwen3VLTextConfig]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLTextConfig</name><anchor>transformers.Qwen3VLTextConfig</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py#L63</source><parameters>[{"name": "vocab_size", "val": " = 151936"}, {"name": "hidden_size", "val": " = 4096"}, {"name": "intermediate_size", "val": " = 22016"}, {"name": "num_hidden_layers", "val": " = 32"}, {"name": "num_attention_heads", "val": " = 32"}, {"name": "num_key_value_heads", "val": " = 32"}, {"name": "head_dim", "val": " = 128"}, {"name": "hidden_act", "val": " = 'silu'"}, {"name": "max_position_embeddings", "val": " = 128000"}, {"name": "initializer_range", "val": " = 0.02"}, {"name": "rms_norm_eps", "val": " = 1e-06"}, {"name": "use_cache", "val": " = True"}, {"name": "tie_word_embeddings", "val": " = False"}, {"name": "rope_theta", "val": " = 5000000.0"}, {"name": "rope_scaling", "val": " = None"}, {"name": "attention_bias", "val": " = False"}, {"name": "attention_dropout", "val": " = 0.0"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **vocab_size** (`int`, *optional*, defaults to 151936) --
  Vocabulary size of the Qwen3VL model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Qwen3VLModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLModel)
- **hidden_size** (`int`, *optional*, defaults to 4096) --
  Dimension of the hidden representations.
- **intermediate_size** (`int`, *optional*, defaults to 22016) --
  Dimension of the MLP representations.
- **num_hidden_layers** (`int`, *optional*, defaults to 32) --
  Number of hidden layers in the Transformer encoder.
- **num_attention_heads** (`int`, *optional*, defaults to 32) --
  Number of attention heads for each attention layer in the Transformer encoder.
- **num_key_value_heads** (`int`, *optional*, defaults to 32) --
  This is the number of key_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
- **head_dim** (`int`, *optional*, defaults to 128) --
  The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
- **hidden_act** (`str` or `function`, *optional*, defaults to `"silu"`) --
  The non-linear activation function (function or string) in the decoder.
- **max_position_embeddings** (`int`, *optional*, defaults to 128000) --
  The maximum sequence length that this model might ever be used with.
- **initializer_range** (`float`, *optional*, defaults to 0.02) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
- **rms_norm_eps** (`float`, *optional*, defaults to 1e-06) --
  The epsilon used by the rms normalization layers.
- **use_cache** (`bool`, *optional*, defaults to `True`) --
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
- **tie_word_embeddings** (`bool`, *optional*, defaults to `False`) --
  Whether the model's input and output word embeddings should be tied.
- **rope_theta** (`float`, *optional*, defaults to 5000000.0) --
  The base period of the RoPE embeddings.
- **rope_scaling** (`Dict`, *optional*) --
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
  'llama3'], with 'default' being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *
  original maximum pre-trained length.
  `original_max_position_embeddings` (`int`, *optional*):
  Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`, *optional*):
  Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`, *optional*):
  Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`, *optional*):
  Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`list[float]`, *optional*):
  Only used with 'longrope'. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`list[float]`, *optional*):
  Only used with 'longrope'. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`, *optional*):
  Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`, *optional*):
  Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
- **attention_bias** (`bool`, defaults to `False`, *optional*, defaults to `False`) --
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
- **attention_dropout** (`float`, *optional*, defaults to 0.0) --
  The dropout ratio for the attention probabilities.</paramsdesc><paramgroups>0</paramgroups></docstring>

This is the configuration class to store the configuration of a [Qwen3VLTextModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLTextModel). It is used to instantiate a
Qwen3-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen3-VL-4B-Instruct [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig) for more information.



<ExampleCodeBlock anchor="transformers.Qwen3VLTextConfig.example">

```python
>>> from transformers import Qwen3VLTextModel, Qwen3VLTextConfig

>>> # Initializing a Qwen3VL style configuration
>>> configuration = Qwen3VLTextConfig()

>>> # Initializing a model from the Qwen3-VL-7B style configuration
>>> model = Qwen3VLTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

</ExampleCodeBlock>

</div>

## Qwen3VLProcessor[[transformers.Qwen3VLProcessor]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLProcessor</name><anchor>transformers.Qwen3VLProcessor</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L62</source><parameters>[{"name": "image_processor", "val": " = None"}, {"name": "tokenizer", "val": " = None"}, {"name": "video_processor", "val": " = None"}, {"name": "chat_template", "val": " = None"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **image_processor** ([Qwen2VLImageProcessor](/docs/transformers/v4.57.1/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), *optional*) --
  The image processor is a required input.
- **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.57.1/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) --
  The tokenizer is a required input.
- **video_processor** ([Qwen3VLVideoProcessor](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor), *optional*) --
  The video processor is a required input.
- **chat_template** (`str`, *optional*) -- A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.</paramsdesc><paramgroups>0</paramgroups></docstring>

Constructs a Qwen3VL processor which wraps a Qwen3VL image processor and a Qwen2 tokenizer into a single processor.
[Qwen3VLProcessor](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v4.57.1/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) and [Qwen2TokenizerFast](/docs/transformers/v4.57.1/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.57.1/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.




<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>post_process_image_text_to_text</name><anchor>transformers.Qwen3VLProcessor.post_process_image_text_to_text</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L287</source><parameters>[{"name": "generated_outputs", "val": ""}, {"name": "skip_special_tokens", "val": " = True"}, {"name": "clean_up_tokenization_spaces", "val": " = False"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **generated_outputs** (`torch.Tensor` or `np.ndarray`) --
  The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
  or `(sequence_length,)`.
- **skip_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `False`) --
  Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
- ****kwargs** --
  Additional arguments to be passed to the tokenizer's `batch_decode method`.</paramsdesc><paramgroups>0</paramgroups><rettype>`list[str]`</rettype><retdesc>The decoded text.</retdesc></docstring>

Post-process the output of the model to decode the text.








</div></div>

## Qwen3VLVideoProcessor[[transformers.Qwen3VLVideoProcessor]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLVideoProcessor</name><anchor>transformers.Qwen3VLVideoProcessor</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py#L87</source><parameters>[{"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessorInitKwargs]"}]</parameters><paramsdesc>- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
- **size** (`dict`, *optional*, defaults to `self.size`) --
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
- **size_divisor** (`int`, *optional*, defaults to `self.size_divisor`) --
  The size by which to make sure both the height and width can be divided.
- **default_to_square** (`bool`, *optional*, defaults to `self.default_to_square`) --
  Whether to default to a square video when resizing, if size is an int.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
- **crop_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) --
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
- **rescale_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) --
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
- **do_convert_rgb** (`bool`, *optional*, defaults to `self.image_std`) --
  Whether to convert the video to RGB.
- **video_metadata** (`VideoMetadata`, *optional*) --
  Metadata of the video containing information about total duration, fps and total number of frames.
- **do_sample_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) --
  Whether to sample frames from the video before processing or to process the whole video.
- **num_frames** (`int`, *optional*, defaults to `self.num_frames`) --
  Maximum number of frames to sample when `do_sample_frames=True`.
- **fps** (`int` or `float`, *optional*, defaults to `self.fps`) --
  Target frames to sample per second when `do_sample_frames=True`.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output video. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
  - Unset: Use the channel dimension format of the input video.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
- **device** (`torch.device`, *optional*) --
  The device to process the videos on. If unset, the device is inferred from the input videos.
- **return_metadata** (`bool`, *optional*) --
  Whether to return video metadata or not.

- **patch_size** (`int`, *optional*, defaults to 16) --
  The spacial patch size of the vision encoder.
- **temporal_patch_size** (`int`, *optional*, defaults to 2) --
  The temporal patch size of the vision encoder.
- **merge_size** (`int`, *optional*, defaults to 2) --
  The merge size of the vision encoder to llm encoder.</paramsdesc><paramgroups>0</paramgroups></docstring>
Constructs a fast Qwen3-VL image processor that dynamically resizes videos based on the original videos.




<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>sample_frames</name><anchor>transformers.Qwen3VLVideoProcessor.sample_frames</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py#L127</source><parameters>[{"name": "metadata", "val": ": VideoMetadata"}, {"name": "num_frames", "val": ": typing.Optional[int] = None"}, {"name": "fps", "val": ": typing.Union[int, float, NoneType] = None"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **video** (`torch.Tensor`) --
  Video that need to be sampled.
- **metadata** (`VideoMetadata`) --
  Metadata of the video containing information about total duration, fps and total number of frames.
- **num_frames** (`int`, *optional*) --
  Maximum number of frames to sample. Defaults to `self.num_frames`.
- **fps** (`int` or `float`, *optional*) --
  Target frames to sample per second. Defaults to `self.fps`.</paramsdesc><paramgroups>0</paramgroups><rettype>torch.Tensor</rettype><retdesc>Sampled video frames.</retdesc></docstring>

Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
and `fps` are mutually exclusive.








</div></div>

## Qwen3VLVisionModel[[transformers.Qwen3VLVisionModel]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLVisionModel</name><anchor>transformers.Qwen3VLVisionModel</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L564</source><parameters>[{"name": "config", "val": ""}, {"name": "*inputs", "val": ""}, {"name": "**kwargs", "val": ""}]</parameters></docstring>



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>forward</name><anchor>transformers.Qwen3VLVisionModel.forward</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L703</source><parameters>[{"name": "hidden_states", "val": ": Tensor"}, {"name": "grid_thw", "val": ": Tensor"}, {"name": "**kwargs", "val": ""}]</parameters><paramsdesc>- **hidden_states** (`torch.Tensor` of shape `(seq_len, hidden_size)`) --
  The final hidden states of the model.
- **grid_thw** (`torch.Tensor` of shape `(num_images_or_videos, 3)`) --
  The temporal, height and width of feature shape of each image in LLM.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.Tensor`</rettype><retdesc>hidden_states.</retdesc></docstring>








</div></div>

## Qwen3VLTextModel[[transformers.Qwen3VLTextModel]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLTextModel</name><anchor>transformers.Qwen3VLTextModel</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L762</source><parameters>[{"name": "config", "val": ": Qwen3VLTextConfig"}]</parameters><paramsdesc>- **config** ([Qwen3VLTextConfig](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLTextConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.</paramsdesc><paramgroups>0</paramgroups></docstring>
Text part of Qwen3VL, not a pure text-only model, as DeepStack integrates visual features into the early hidden states.

This model inherits from [PreTrainedModel](/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.





<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>forward</name><anchor>transformers.Qwen3VLTextModel.forward</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L782</source><parameters>[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "visual_pos_masks", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "deepstack_visual_embeds", "val": ": typing.Optional[list[torch.Tensor]] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}]</parameters><paramsdesc>- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.57.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
- **visual_pos_masks** (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*) --
  The mask of the visual positions.
- **deepstack_visual_embeds** (`list[torch.Tensor]`, *optional*) --
  The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
  The feature is extracted from the different visual encoder layers, and fed to the decoder
  hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).</paramsdesc><paramgroups>0</paramgroups><rettype>[transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/v4.57.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`</rettype><retdesc>A [transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/v4.57.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3VLConfig](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.</retdesc></docstring>
The [Qwen3VLTextModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLTextModel) forward method, overrides the `__call__` special method.

<Tip>

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

</Tip>








</div></div>

## Qwen3VLModel[[transformers.Qwen3VLModel]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLModel</name><anchor>transformers.Qwen3VLModel</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L887</source><parameters>[{"name": "config", "val": ""}]</parameters><paramsdesc>- **config** ([Qwen3VLModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLModel)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.</paramsdesc><paramgroups>0</paramgroups></docstring>

The bare Qwen3 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.





<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>forward</name><anchor>transformers.Qwen3VLModel.forward</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1106</source><parameters>[{"name": "input_ids", "val": ": LongTensor = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "pixel_values_videos", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_grid_thw", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "video_grid_thw", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]</parameters><paramsdesc>- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.57.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
- **pixel_values_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) --
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses
  `video_processor_class` for processing videos).
- **image_grid_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) --
  The temporal, height and width of feature shape of each image in LLM.
- **video_grid_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) --
  The temporal, height and width of feature shape of each video in LLM.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.</paramsdesc><paramgroups>0</paramgroups><rettype>`transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModelOutputWithPast` or `tuple(torch.FloatTensor)`</rettype><retdesc>A `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **rope_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) -- The rope index difference between sequence length and multimodal rope.</retdesc></docstring>
The [Qwen3VLModel](/docs/transformers/v4.57.1/en/model_doc/qwen3_vl#transformers.Qwen3VLModel) forward method, overrides the `__call__` special method.

<Tip>

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

</Tip>








</div></div>

## Qwen3VLForConditionalGeneration[[transformers.Qwen3VLForConditionalGeneration]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.Qwen3VLForConditionalGeneration</name><anchor>transformers.Qwen3VLForConditionalGeneration</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1271</source><parameters>[{"name": "config", "val": ""}]</parameters></docstring>



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>forward</name><anchor>transformers.Qwen3VLForConditionalGeneration.forward</anchor><source>https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1314</source><parameters>[{"name": "input_ids", "val": ": LongTensor = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "pixel_values_videos", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_grid_thw", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "video_grid_thw", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]</parameters></docstring>

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
The temporal, height and width of feature shape of each image in LLM.
video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
The temporal, height and width of feature shape of each video in LLM.

Example:
TODO: Add example


</div></div>

<EditOnGithub source="https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_vl.md" />