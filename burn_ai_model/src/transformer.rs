use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::{
    module::Param,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    },
};

#[derive(Config, Debug)]
pub struct BattleModelConfig {
    #[config(default = 64)]
    pub d_model: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub num_classes: usize,
    #[config(default = 22)]
    pub tile_features: usize,
    #[config(default = 4)]
    pub meta_features: usize,
    #[config(default = 11)]
    pub grid_size: usize,
}

#[derive(Module, Debug)]
pub struct BattleModel<B: Backend> {
    tile_projection: Linear<B>,
    pos_projection: Linear<B>,

    // NEW: Project global stats (4 floats) -> Embedding (64 floats)
    meta_projection: Linear<B>,

    cls_token: Param<Tensor<B, 2>>,
    transformer: TransformerEncoder<B>,
    output_head: Linear<B>,

    grid_size: usize,
}

impl<B: Backend> BattleModel<B> {
    pub fn new(config: &BattleModelConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;

        let tile_projection = LinearConfig::new(config.tile_features, d_model).init(device);
        let pos_projection = LinearConfig::new(2, d_model).init(device);

        // NEW: Init the Meta Projector
        let meta_projection = LinearConfig::new(config.meta_features, d_model).init(device);

        let cls_token = Param::from_tensor(Tensor::random(
            [1, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        ));

        let transformer =
            TransformerEncoderConfig::new(d_model, config.d_ff, config.n_heads, config.n_layers)
                .init(device);

        let output_head = LinearConfig::new(d_model, config.num_classes).init(device);

        Self {
            tile_projection,
            pos_projection,
            meta_projection,
            cls_token,
            transformer,
            output_head,
            grid_size: config.grid_size,
        }
    }

    // UPDATED Signature: Now accepts TWO tensors
    pub fn forward(&self, tiles: Tensor<B, 3>, metadata: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, _, _] = tiles.dims();
        let device = tiles.device();

        // 1. Embed Tiles [Batch, 121, 64]
        let x_tiles = self.tile_projection.forward(tiles);
        let pos_grid = self.generate_pos_grid(batch_size, &device);
        let pos_embeds = self.pos_projection.forward(pos_grid);
        let x_tiles = x_tiles + pos_embeds;

        // 2. Embed Metadata [Batch, 4] -> [Batch, 64]
        let x_meta = self.meta_projection.forward(metadata);

        // Reshape for sequence: [Batch, 64] -> [Batch, 1, 64]
        let x_meta = x_meta.unsqueeze_dim(1);

        // 3. Prepare CLS [Batch, 1, 64]
        let x_cls = self
            .cls_token
            .val()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // 4. Concatenate: [CLS] + [META] + [TILES]
        // Result shape: [Batch, 1 + 1 + 121, 64] => [Batch, 123, 64]
        let seq = Tensor::cat(vec![x_cls, x_meta, x_tiles], 1);

        // 5. Transformer
        let encoded = self.transformer.forward(TransformerEncoderInput::new(seq));

        // 6. Output (Take Index 0, the CLS token)
        let cls_out = encoded.slice([0..batch_size, 0..1]).squeeze_dim(1);

        self.output_head.forward(cls_out)
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
