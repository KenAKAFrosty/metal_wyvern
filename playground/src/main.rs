// use burn_import::onnx::ModelGen;

// fn main() {
//     ModelGen::new()
//         .input("simple_cnn_opset16.onnx")
//         .out_dir("src/model/")
//         .run_from_cli();
// }

fn main() {
    println!("Placeholder. See commented-out code for model generation");
}

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
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 1024)]
    pub d_ff: usize, // Inner dimension of feed-forward network
    #[config(default = 4)] // UP, RIGHT, DOWN, LEFT
    pub num_classes: usize,
    #[config(default = 22)] // The 22-float input vector we designed
    pub input_features: usize,
    #[config(default = 11)] // 11x11 board
    pub grid_size: usize,
}

#[derive(Module, Debug)]
pub struct BattleModel<B: Backend> {
    // 1. The "Mixer" - Projects raw features to latent space
    tile_projection: Linear<B>,

    // 2. The "GPS" - Projects (x,y) coords to latent space
    pos_projection: Linear<B>,

    // 3. The "[CLS]" Token - A learnable parameter
    cls_token: Param<Tensor<B, 2>>,

    // 4. The Brain - Standard Transformer Encoder
    transformer: TransformerEncoder<B>,

    // 5. The Output Head
    output_head: Linear<B>,

    // Config cache for shape info
    grid_size: usize,
    d_model: usize,
}

impl<B: Backend> BattleModel<B> {
    pub fn new(config: &BattleModelConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;

        // Projection: 22 raw floats -> 64 dim embedding
        let tile_projection = LinearConfig::new(config.input_features, d_model).init(device);

        // Positional: 2 coords (x,y) -> 64 dim embedding
        let pos_projection = LinearConfig::new(2, d_model).init(device);

        // CLS Token: Shape [1, d_model]
        // We initialize it randomly. It will learn during training.
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
            cls_token,
            transformer,
            output_head,
            grid_size: config.grid_size,
            d_model,
        }
    }

    /// Forward pass
    /// input: [Batch, 121, 22] - The flattened board features
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, seq_len, _] = input.dims();
        let device = input.device();

        // 1. Embed the Tiles
        // [Batch, 121, 22] -> [Batch, 121, 64]
        let x = self.tile_projection.forward(input);

        // 2. Add Positional Encodings
        // We generate the grid coordinates on the fly (cheap on CPU)
        // This makes the model size-agnostic if we change grid_size later
        let pos_grid = self.generate_pos_grid(batch_size, &device); // [Batch, 121, 2]
        let pos_embeds = self.pos_projection.forward(pos_grid); // [Batch, 121, 64]

        // Add them up (Broadcasting might happen, but usually explicit match is safer)
        let x = x + pos_embeds;

        // 3. Prepend [CLS] token
        // Expand CLS to batch size: [1, 64] -> [Batch, 1, 64]
        let cls_expanded = self
            .cls_token
            .val()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        // Concatenate: [Batch, 1, 64] + [Batch, 121, 64] -> [Batch, 122, 64]
        let x = Tensor::cat(vec![cls_expanded, x], 1);

        // 4. Run Transformer
        // Burn 0.19 TransformerInput expects a tensor and optional mask
        let trans_out = self.transformer.forward(TransformerEncoderInput::new(x));

        // 5. Extract CLS (Index 0)
        // slice(0..1) keeps dim, so [Batch, 1, 64]
        let cls_out = trans_out.slice([0..batch_size, 0..1]);

        // Flatten to [Batch, 64]
        let cls_out = cls_out.squeeze_dim(1);

        // 6. Classification
        self.output_head.forward(cls_out)
    }

    /// Helper to create normalized (x,y) grid
    /// Returns tensor of shape [Batch, 121, 2]
    fn generate_pos_grid(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        let size = self.grid_size;
        let mut coords = Vec::with_capacity(size * size * 2);

        for y in 0..size {
            for x in 0..size {
                // Normalize to 0.0 - 1.0 range
                coords.push(x as f32 / (size - 1) as f32);
                coords.push(y as f32 / (size - 1) as f32);
            }
        }

        // Create [121, 2] tensor
        let grid = Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::new(coords, vec![size * size, 2]),
            device,
        );

        // Expand to batch: [121, 2] -> [Batch, 121, 2]
        grid.unsqueeze_dim(0).repeat_dim(0, batch_size)
    }
}
