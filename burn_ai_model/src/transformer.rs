use burn::nn::{
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::relu;

#[derive(Config, Debug)]
pub struct BattleModelConfig {
    #[config(default = 64)] // Keep small for speed (Matrix multiplication cost)
    pub d_model: usize,
    #[config(default = 128)] // Internal brain of the transformer
    pub d_ff: usize,
    #[config(default = 4)] // 64 / 4 = 16 dim per head (Minimum viable)
    pub n_heads: usize,
    #[config(default = 6)] // Deep reasoning layers
    pub n_layers: usize,
    #[config(default = 4)] // UP, RIGHT, DOWN, LEFT
    pub num_classes: usize,
    #[config(default = 22)] // Input features per tile
    pub tile_features: usize,
    #[config(default = 2)] // Input global metadata features
    pub meta_features: usize,
    #[config(default = 11)] // Standard board size
    pub grid_size: usize,

    #[config(default = 256)]
    pub head_compress_size: usize,
    #[config(default = 1024)]
    pub head_expand_size: usize,
}

#[derive(Module, Debug)]
pub struct BattleModel<B: Backend> {
    tile_projection: Linear<B>,
    pos_projection: Linear<B>,
    meta_projection: Linear<B>,

    transformer: TransformerEncoder<B>,

    // The Funnel MLP Head
    head_compress: Linear<B>, // Step 1: Compress huge flat vector
    head_expand: Linear<B>,   // Step 2: Expand to strategy space
    head_output: Linear<B>,   // Step 3: Decision

    grid_size: usize,
    d_model: usize,
}

impl<B: Backend> BattleModel<B> {
    pub fn new(config: &BattleModelConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;

        // 1. Projections
        let tile_projection = LinearConfig::new(config.tile_features, d_model).init(device);
        let pos_projection = LinearConfig::new(2, d_model).init(device);
        let meta_projection = LinearConfig::new(config.meta_features, d_model).init(device);

        // 2. Transformer (Eyes)
        let transformer =
            TransformerEncoderConfig::new(d_model, config.d_ff, config.n_heads, config.n_layers)
                .init(device);

        // 3. Calculate Flattened Input Size
        // Grid (11*11) * d_model (64) = 7744
        let flat_board_size = config.grid_size * config.grid_size * d_model;

        // We will concatenate the Metadata Embedding (size 64) to this
        let mlp_input_size = flat_board_size + d_model;

        // 4. The Funnel Head (Brain)
        // Stage A: Compress (7808 -> 256)
        let head_compress =
            LinearConfig::new(mlp_input_size, config.head_compress_size).init(device);
        // Stage B: Strategy Expand (256 -> 1024)
        let head_expand =
            LinearConfig::new(config.head_compress_size, config.head_expand_size).init(device);
        // Stage C: Output (1024 -> 4)
        let head_output =
            LinearConfig::new(config.head_expand_size, config.num_classes).init(device);

        Self {
            tile_projection,
            pos_projection,
            meta_projection,
            transformer,
            head_compress,
            head_expand,
            head_output,
            grid_size: config.grid_size,
            d_model,
        }
    }

    pub fn forward(&self, tiles: Tensor<B, 3>, metadata: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, _, _] = tiles.dims();
        let device = tiles.device();

        // 1. Embed Tiles [Batch, 121, 64]
        let x_tiles = self.tile_projection.forward(tiles);
        let pos_grid = self.generate_pos_grid(batch_size, &device);
        let pos_embeds = self.pos_projection.forward(pos_grid);

        // Combine Content + Location
        let x = x_tiles + pos_embeds;

        // 2. Run Transformer
        // Note: No CLS token, no Metadata in the sequence. Just the board.
        let encoded = self.transformer.forward(TransformerEncoderInput::new(x));

        // [Batch, 121, 64] -> [Batch, 7744]
        // Merge dimension 1 (121) through dimension 2 (64)
        let flattened_board = encoded.flatten(1, 2);

        // 4. Embed Metadata [Batch, 2] -> [Batch, 64]
        let meta_embed = self.meta_projection.forward(metadata);

        // 5. Concatenate [Board (7744) + Meta (64)] -> [Batch, 7808]
        let mlp_input = Tensor::cat(vec![flattened_board, meta_embed], 1);

        // 6. MLP Funnel
        let h = self.head_compress.forward(mlp_input);
        let h = relu(h);

        let h = self.head_expand.forward(h);
        let h = relu(h);

        self.head_output.forward(h)
    }

    fn generate_pos_grid(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        let size = self.grid_size;
        let mut coords = Vec::with_capacity(size * size * 2);
        for y in 0..size {
            for x in 0..size {
                coords.push(x as f32 / (size - 1) as f32);
                coords.push(y as f32 / (size - 1) as f32);
            }
        }
        let grid = Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::new(coords, vec![size * size, 2]),
            device,
        );
        grid.unsqueeze_dim(0).repeat_dim(0, batch_size)
    }
}
