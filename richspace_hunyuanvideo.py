class HyVideoRichSpaceTextEncode:
    """
    RichSpace-enabled text encoder that maintains compatibility with
    HunyuanVideo's 3D attention architecture.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_encoders": ("HYVIDTEXTENCODER",),
                "prompt_a": ("STRING", {"default": "", "multiline": True}),
                "prompt_b": ("STRING", {"default": "", "multiline": True}),
                "guide_prompt": ("STRING", {"default": "", "multiline": True}),
                "interpolation_steps": ("INT", {"default": 30, "min": 2, "max": 100})
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "prompt_template": (["video", "image", "custom", "disabled"],),
                "custom_prompt_template": ("PROMPT_TEMPLATE",),
                "clip_l": ("CLIP",),
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS",)
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, text_encoders, prompt_a, prompt_b, guide_prompt,
                interpolation_steps=30, force_offload=True, prompt_template="video",
                custom_prompt_template=None, clip_l=None):
        """
        Process prompts using RichSpace interpolation while maintaining
        compatibility with HunyuanVideo's 3D attention architecture.
        """
        device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        # Setup text encoder
        text_encoder_1 = text_encoders["text_encoder"]
        text_encoder_1.to(device)
        
        # Handle prompt templates for proper HunyuanVideo formatting
        if prompt_template == "video":
            prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode-video"]
        elif prompt_template == "custom":
            prompt_template_dict = custom_prompt_template
        else:
            prompt_template_dict = None

        def get_embedding_with_mask(prompt):
            """Get embedding while preserving attention structure"""
            text_inputs = text_encoder_1.text2tokens(prompt, 
                                                   prompt_template=prompt_template_dict)
            outputs = text_encoder_1.encode(text_inputs, 
                                          prompt_template=prompt_template_dict,
                                          device=device)
            return outputs.hidden_state, outputs.attention_mask

        # Process embeddings while maintaining 3D attention compatibility
        interpolator = RichSpaceInterpolator(text_encoder_1)
        
        with torch.autocast(device_type=mm.get_autocast_device(device), 
                          dtype=text_encoder_1.dtype, 
                          enabled=text_encoder_1.is_fp8):
            # Get base embeddings
            embed_a, mask_a = get_embedding_with_mask(prompt_a)
            embed_b, mask_b = get_embedding_with_mask(prompt_b)
            embed_guide, mask_guide = get_embedding_with_mask(guide_prompt)
            
            # Find optimal interpolation
            optimal_embedding, optimal_mask, step_idx = interpolator.find_optimal_interpolation(
                embed_a, embed_b, embed_guide,
                mask_a, mask_b, mask_guide,
                interpolation_steps
            )
            
            log.info(f"Using RichSpace interpolation step {step_idx}/{interpolation_steps}")
            
            # Handle secondary encoder if present
            prompt_embeds_2 = None
            attention_mask_2 = None
            if clip_l is not None:
                clip_l.cond_stage_model.to(device)
                tokens = clip_l.tokenize(guide_prompt, return_word_ids=True)
                prompt_embeds_2 = clip_l.encode_from_tokens(tokens, return_pooled=True, 
                                                          return_dict=False)[1]
                if force_offload:
                    clip_l.cond_stage_model.to(offload_device)

        # Handle device offloading
        if force_offload:
            text_encoder_1.to(offload_device)
            mm.soft_empty_cache()

        # Construct embeddings dictionary that matches HunyuanVideo's expectations
        prompt_embeds_dict = {
            "prompt_embeds": optimal_embedding,
            "attention_mask": optimal_mask,
            "negative_prompt_embeds": None,
            "negative_attention_mask": None,
            "prompt_embeds_2": prompt_embeds_2,
            "attention_mask_2": attention_mask_2,
            "negative_prompt_embeds_2": None,
            "negative_attention_mask_2": None,
        }
        
        return (prompt_embeds_dict,)
