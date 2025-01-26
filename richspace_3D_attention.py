class RichSpaceInterpolator:
    """
    Handles text embedding interpolation following the RichSpace algorithms while maintaining
    compatibility with HunyuanVideo's 3D attention architecture.
    """
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def calculate_perpendicular_foot(self, embed_a: torch.Tensor, embed_b: torch.Tensor, 
                                   embed_guide: torch.Tensor) -> torch.Tensor:
        """
        Calculates the perpendicular foot point as described in Algorithm 3.
        This projection helps find the optimal point between embeddings that
        best captures the guide prompt's intent.
        
        The calculation preserves the embedding dimensions needed for 3D attention:
        embed shape: [n × d] where n is sequence length and d is hidden dimension
        """
        device = embed_a.device
        dtype = embed_a.dtype
        
        # First calculate the projection vectors while preserving dimensions
        vec_ac = embed_guide - embed_a  # Guide direction
        vec_ab = embed_b - embed_a      # Interpolation direction
        
        # Calculate projection while maintaining attention-compatible shape
        # Use proper broadcasting for sequence-level operations
        proj_length = torch.sum(vec_ac * vec_ab, dim=-1, keepdim=True) / \
                     (torch.sum(vec_ab * vec_ab, dim=-1, keepdim=True) + 1e-8)
        
        # Calculate projection while preserving sequence structure
        proj_vec = proj_length.unsqueeze(-1) * vec_ab
        foot_point = embed_a + proj_vec
        
        return foot_point.to(dtype=dtype, device=device)

    def find_optimal_interpolation(self, embed_a: torch.Tensor, embed_b: torch.Tensor, 
                                 embed_guide: torch.Tensor, 
                                 attention_mask_a: torch.Tensor,
                                 attention_mask_b: torch.Tensor,
                                 attention_mask_guide: torch.Tensor,
                                 num_steps: int = 30) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Finds optimal interpolation following Algorithm 2, while maintaining
        compatibility with HunyuanVideo's attention mechanisms.
        
        Returns:
            Tuple of (optimal embedding, optimal attention mask, step index)
        """
        foot_point = self.calculate_perpendicular_foot(embed_a, embed_b, embed_guide)
        
        similarities = []
        interpolated_embeds = []
        interpolated_masks = []
        
        # Generate interpolations while preserving attention structure
        for i in range(num_steps):
            t = i / (num_steps - 1)
            # Interpolate embeddings
            interp_embed = t * embed_a + (1-t) * embed_b
            # Interpolate attention masks (using ceil to preserve attention)
            interp_mask = torch.ceil(t * attention_mask_a + (1-t) * attention_mask_b)
            
            # Calculate similarity while respecting attention mask
            masked_interp = interp_embed * interp_mask.unsqueeze(-1)
            masked_foot = foot_point * attention_mask_guide.unsqueeze(-1)
            
            sim = F.cosine_similarity(
                masked_interp.reshape(1, -1),
                masked_foot.reshape(1, -1)
            )
            
            similarities.append(sim.item())
            interpolated_embeds.append(interp_embed)
            interpolated_masks.append(interp_mask)

        optimal_idx = torch.argmax(torch.tensor(similarities))
        return (interpolated_embeds[optimal_idx], 
                interpolated_masks[optimal_idx], 
                optimal_idx)
