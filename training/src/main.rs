#![recursion_limit = "256"]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use burn_ai_model::transformer::{BattleModel, BattleModelConfig};

// ------------------------------------------------------------------
// 1. Data Structures (Unchanged)
// ------------------------------------------------------------------
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct DeathInfo {
    turn: u32,
    cause: String,
    eliminated_by: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Snake {
    #[serde(rename = "ID")]
    id: String,
    body: Vec<Position>,
    health: i32,
    is_bot: bool,
    death: Option<DeathInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct Ruleset {
    food_spawn_chance: String,
    minimum_food: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
struct Game {
    width: u32,
    height: u32,
    ruleset: Ruleset,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TrainingExample {
    #[serde(rename = "Game")]
    game: Game,
    #[serde(rename = "Snakes")]
    snakes: Vec<Snake>,
    #[serde(rename = "Food")]
    food: Vec<Position>,
    #[serde(rename = "Hazards")]
    hazards: Vec<Position>,
    #[serde(rename = "Turn")]
    turn: u32,
    label: usize,
    winning_snake_id: String,
}

// ------------------------------------------------------------------
// 2. Dataset (Unchanged)
// ------------------------------------------------------------------
#[derive(Clone)]
struct BattlesnakeDataset {
    items: Vec<TrainingExample>,
}

impl BattlesnakeDataset {
    pub fn new(items: Vec<TrainingExample>) -> Self {
        Self { items }
    }
}

impl Dataset<TrainingExample> for BattlesnakeDataset {
    fn get(&self, index: usize) -> Option<TrainingExample> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

fn load_data(db_path: &str) -> Vec<TrainingExample> {
    println!("Loading data from {}...", db_path);
    let conn = Connection::open(db_path).expect("Could not open DB");
    let mut stmt = conn
        .prepare("SELECT data_json FROM training_examples")
        .expect("Table not found");
    let items_iter = stmt
        .query_map([], |row| {
            let json: String = row.get(0)?;
            Ok(serde_json::from_str::<TrainingExample>(&json).unwrap())
        })
        .expect("Query failed");

    let mut items = Vec::new();
    for item in items_iter {
        if let Ok(ex) = item {
            items.push(ex);
        }
    }
    println!("Total examples loaded: {}", items.len());
    items
}

// ------------------------------------------------------------------
// 3. Batcher (THE BIG REWRITE)
// ------------------------------------------------------------------

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Clone)]
struct BattlesnakeBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct BattlesnakeBatch<B: Backend> {
    // [Batch, 121, 22] - The sequence of tiles
    tiles: Tensor<B, 3>,
    // [Batch, 4] - The global metadata
    metadata: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

const GRID_SIZE: usize = 11;
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE; // 121
const TILE_FEATS: usize = 22;
const META_FEATS: usize = 4;

// Helper to set features in the flat vector
// grid: flat vector of size 121 * 22
// idx: 0..120 (tile index)
// feat: 0..21 (feature index)
// val: value to set
fn set_feat(grid: &mut [f32], x: i32, y: i32, feat: usize, val: f32) {
    if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
        let tile_idx = (y as usize * GRID_SIZE) + x as usize;
        let vec_idx = (tile_idx * TILE_FEATS) + feat;
        grid[vec_idx] = val;
    }
}

impl<B: Backend> Batcher<B, TrainingExample, BattlesnakeBatch<B>> for BattlesnakeBatcher<B> {
    fn batch(&self, items: Vec<TrainingExample>, device: &B::Device) -> BattlesnakeBatch<B> {
        let batch_size = items.len();

        let mut tiles_data = Vec::with_capacity(batch_size * SEQ_LEN * TILE_FEATS);
        let mut meta_data = Vec::with_capacity(batch_size * META_FEATS);
        let mut targets_data = Vec::with_capacity(batch_size);

        for item in items {
            let w = item.game.width as usize;
            let h = item.game.height as usize;
            let area = (w * h) as f32;

            // 1. Prepare Tile Grid (Flat buffer)
            let mut grid = vec![0.0f32; SEQ_LEN * TILE_FEATS];

            // Identify "Me" and "Them"
            let target_id = &item.winning_snake_id; // We train to mimic the winner
            let mut my_snake: Option<&Snake> = None;
            let mut enemies: Vec<&Snake> = Vec::new();

            for snake in &item.snakes {
                if snake.death.is_some() {
                    continue;
                } // Skip dead
                if snake.id == *target_id {
                    my_snake = Some(snake);
                } else {
                    enemies.push(snake);
                }
            }

            // --- Fill "Me" Features (Indices 0-4) ---
            if let Some(me) = my_snake {
                let norm_health = me.health as f32 / 100.0;
                let norm_len = me.body.len() as f32 / area;

                // Head (Idx 0)
                if !me.body.is_empty() {
                    set_feat(&mut grid, me.body[0].x, me.body[0].y, 0, 1.0);
                    // Holographic stats on Head
                    set_feat(&mut grid, me.body[0].x, me.body[0].y, 3, norm_health);
                    set_feat(&mut grid, me.body[0].x, me.body[0].y, 4, norm_len);
                }

                // Body (Idx 1) & Tail (Idx 2)
                for (i, part) in me.body.iter().enumerate().skip(1) {
                    let is_tail = i == me.body.len() - 1;
                    let feat_idx = if is_tail { 2 } else { 1 };

                    set_feat(&mut grid, part.x, part.y, feat_idx, 1.0);
                    // Holographic stats on Body parts too!
                    set_feat(&mut grid, part.x, part.y, 3, norm_health);
                    set_feat(&mut grid, part.x, part.y, 4, norm_len);
                }
            }

            // --- Fill "Enemy" Features (Indices 5-19) ---
            // We support up to 3 enemies.
            // Sort enemies by ID to ensure consistency across frames
            enemies.sort_by_key(|s| &s.id);

            for (i, enemy) in enemies.iter().take(3).enumerate() {
                let offset = 5 + (i * 5); // 5, 10, 15
                let norm_health = enemy.health as f32 / 100.0;
                let norm_len = enemy.body.len() as f32 / area;

                // Head
                if !enemy.body.is_empty() {
                    set_feat(&mut grid, enemy.body[0].x, enemy.body[0].y, offset + 0, 1.0);
                    set_feat(
                        &mut grid,
                        enemy.body[0].x,
                        enemy.body[0].y,
                        offset + 3,
                        norm_health,
                    );
                    set_feat(
                        &mut grid,
                        enemy.body[0].x,
                        enemy.body[0].y,
                        offset + 4,
                        norm_len,
                    );
                }
                // Body/Tail
                for (j, part) in enemy.body.iter().enumerate().skip(1) {
                    let is_tail = j == enemy.body.len() - 1;
                    let feat_idx = if is_tail { offset + 2 } else { offset + 1 };
                    set_feat(&mut grid, part.x, part.y, feat_idx, 1.0);
                    set_feat(&mut grid, part.x, part.y, offset + 3, norm_health);
                    set_feat(&mut grid, part.x, part.y, offset + 4, norm_len);
                }
            }

            // --- Food (Idx 20) ---
            for f in &item.food {
                set_feat(&mut grid, f.x, f.y, 20, 1.0);
            }

            // --- Hazards (Idx 21) ---
            for hz in &item.hazards {
                set_feat(&mut grid, hz.x, hz.y, 21, 1.0);
            }

            tiles_data.extend(grid);

            // 2. Prepare Metadata (4 floats)
            // [Turn, SpawnChance, MinFood, HazardDamage]
            let fs_chance: f32 = item.game.ruleset.food_spawn_chance.parse().unwrap_or(15.0);
            let min_food: f32 = item.game.ruleset.minimum_food.parse().unwrap_or(1.0);

            meta_data.push((item.turn as f32 / 500.0).min(1.0)); // Turn normalized
            meta_data.push(fs_chance / 100.0);
            meta_data.push(min_food / area);
            meta_data.push(0.14); // Assume 14 hazard damage (standard) normalized

            targets_data.push(item.label as i32);
        }

        // Create Tensors
        let tiles_shape = [batch_size, SEQ_LEN, TILE_FEATS];
        let tiles = Tensor::from_data(TensorData::new(tiles_data, tiles_shape), &self.device);

        let meta_shape = [batch_size, META_FEATS];
        let metadata = Tensor::from_data(TensorData::new(meta_data, meta_shape), &self.device);

        let targets = Tensor::from_data(TensorData::new(targets_data, [batch_size]), &self.device);

        BattlesnakeBatch {
            tiles,
            metadata,
            targets,
        }
    }
}

// ------------------------------------------------------------------
// 4. Train Step (Updated signature)
// ------------------------------------------------------------------

impl<B: AutodiffBackend> TrainStep<BattlesnakeBatch<B>, ClassificationOutput<B>>
    for BattleModel<B>
{
    fn step(&self, batch: BattlesnakeBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Forward pass now takes (Tiles, Meta)
        let logits = self.forward(batch.tiles, batch.metadata);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            ClassificationOutput {
                loss,
                output: logits,
                targets: batch.targets,
            },
        )
    }
}

impl<B: Backend> ValidStep<BattlesnakeBatch<B>, ClassificationOutput<B>> for BattleModel<B> {
    fn step(&self, batch: BattlesnakeBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.tiles, batch.metadata);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        ClassificationOutput {
            loss,
            output: logits,
            targets: batch.targets,
        }
    }
}

// ------------------------------------------------------------------
// 5. Main
// ------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let device = WgpuDevice::default();

    // --- HYPERPARAMETERS ---
    let batch_size = 64;
    let learning_rate = 1e-4;
    let num_epochs = 30;

    // Define Model Config
    let config = BattleModelConfig {
        d_model: 256, // Embedding size (Start small)
        d_ff: 1024,   // Feed forward inner dimension
        n_heads: 8,   // Attention heads
        n_layers: 6,  // Transformer layers
        num_classes: 4,
        tile_features: 22, // Match Batcher
        meta_features: 4,  // Match Batcher
        grid_size: 11,
    };

    println!("Initializing Holographic Transformer...");
    println!("Config: {:?}", config);

    // Initialize Model
    let model: BattleModel<MyAutodiffBackend> = BattleModel::new(&config, &device);
    let optimizer = AdamWConfig::new().init();

    // Load Data
    let mut all_data = load_data("../battlesnake_data.db");
    if all_data.len() <= batch_size {
        panic!("Not enough data!");
    }

    // Split Val
    let valid_data = all_data.split_off(all_data.len() - batch_size);
    let train_data = all_data;

    let dataset_train = BattlesnakeDataset::new(train_data);
    let dataset_valid = BattlesnakeDataset::new(valid_data);

    let batcher_train = BattlesnakeBatcher::<MyAutodiffBackend> {
        device: device.clone(),
    };
    let batcher_valid = BattlesnakeBatcher::<MyBackend> {
        device: device.clone(),
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(dataset_valid);

    let artifact_dir = "/tmp/battlesnake-transformer";

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .build(model, optimizer, learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .model
        .save_file("transformer_v1", &CompactRecorder::new())
        .expect("Failed to save model");

    println!("Transformer training complete!");
}
