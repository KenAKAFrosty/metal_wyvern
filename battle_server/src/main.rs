mod model;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use model::simple_cnn_opset16::Model;

// DEFINE THE BACKEND
// We use NdArray for pure CPU execution.
// <f32> indicates the float precision.
type B = NdArray<f32>;

#[derive(Deserialize)]
struct Position {
    x: usize,
    y: usize,
}

#[derive(Deserialize)]
struct Snake {
    id: String,
    health: u32,
    body: Vec<Position>,
    #[serde(default)]
    length: usize,
}

#[derive(Deserialize)]
struct RulesetSettings {
    #[serde(rename = "foodSpawnChance")]
    food_spawn_chance: u32,
    #[serde(rename = "minimumFood")]
    minimum_food: u32,
}

#[derive(Deserialize)]
struct Ruleset {
    settings: RulesetSettings,
}

#[derive(Deserialize)]
struct Game {
    ruleset: Ruleset,
}

#[derive(Deserialize)]
struct Board {
    height: usize,
    width: usize,
    food: Vec<Position>,
    hazards: Vec<Position>,
    snakes: Vec<Snake>,
}

#[derive(Deserialize)]
struct GameMoveRequest {
    game: Game,
    turn: u32,
    board: Board,
    you: Snake,
}

#[derive(Serialize)]
struct MoveResponse {
    r#move: String,
    shout: String,
}

#[derive(Serialize)]
struct InfoResponse {
    apiversion: String,
    color: String,
    head: String,
    tail: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

const CHANNELS: usize = 8;
const HEIGHT: usize = 11;
const WIDTH: usize = 11;
const METADATA_COUNT: usize = 8;

fn preprocess(req: &GameMoveRequest) -> (Array4<f32>, Array2<f32>) {
    let (w, h) = (req.board.width, req.board.height);
    let area = (w * h) as f32;
    let target_id = &req.you.id;

    let mut board = Array4::<f32>::zeros((1, CHANNELS, HEIGHT, WIDTH));

    let mut other_len_sum = 0usize;
    let mut other_count = 0usize;
    let mut longest_other = 0usize;

    for snake in &req.board.snakes {
        let len = snake.body.len();
        let norm_len = len as f32 / area;
        let norm_health = snake.health as f32 / 100.0;
        let is_target = snake.id == *target_id;

        if !is_target {
            other_len_sum += len;
            other_count += 1;
            longest_other = longest_other.max(len);
        }

        // Head
        let head = &snake.body[0];
        let head_ch = if is_target { 0 } else { 1 };
        board[[0, head_ch, head.y, head.x]] = norm_len;

        // Body (excluding head and tail)
        let body_ch = if is_target { 2 } else { 3 };
        for part in snake
            .body
            .iter()
            .skip(1)
            .take(snake.body.len().saturating_sub(2))
        {
            board[[0, body_ch, part.y, part.x]] = norm_health;
        }

        // Tail
        if let Some(tail) = snake.body.last() {
            let tail_ch = if is_target { 4 } else { 5 };
            board[[0, tail_ch, tail.y, tail.x]] = norm_health;
        }
    }

    // Food
    for f in &req.board.food {
        if f.x < w && f.y < h {
            board[[0, 6, f.y, f.x]] += 1.0;
        }
    }

    // Hazards
    for hz in &req.board.hazards {
        if hz.x < w && hz.y < h {
            board[[0, 7, hz.y, hz.x]] += 1.0;
        }
    }

    // Metadata
    let food_spawn = req.game.ruleset.settings.food_spawn_chance as f32 / 100.0;
    let min_food = req.game.ruleset.settings.minimum_food as f32 / area;
    let turn = req.turn as f32 / 7200.0;
    let head_x = req.you.body[0].x as f32 / w as f32;
    let head_y = req.you.body[0].y as f32 / h as f32;
    let health = req.you.health as f32 / 100.0;
    let target_len = req.you.body.len();
    let is_longest = if target_len > longest_other { 1.0 } else { 0.0 };
    let avg_frac = if is_longest > 0.5 && other_count > 0 {
        (other_len_sum as f32 / other_count as f32) / target_len as f32
    } else {
        0.0
    };

    let metadata = Array2::from_shape_vec(
        (1, METADATA_COUNT),
        vec![
            food_spawn, min_food, turn, head_x, head_y, health, is_longest, avg_frac,
        ],
    )
    .unwrap();

    (board, metadata)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn info() -> Json<InfoResponse> {
    Json(InfoResponse {
        apiversion: "1".into(),
        color: "#D34516".into(),
        head: "cute-dragon".into(),
        tail: "dragon".into(),
    })
}

async fn handle_move(
    State(model): State<Arc<Mutex<Model<B>>>>,
    Json(req): Json<GameMoveRequest>,
) -> Json<MoveResponse> {
    let (best_move, shout) = tokio::task::spawn_blocking(move || {
        let (board_ndarray, metadata_ndarray) = preprocess(&req);

        // 1. Extract shape and raw data from ndarray
        // .into_raw_vec() is zero-copy (it consumes the ndarray) and efficient.
        // It works perfectly because your preprocess() creates contiguous arrays.
        let board_shape = board_ndarray.shape().to_vec();
        let board_vec = board_ndarray.into_raw_vec();
        let board_data = TensorData::new(board_vec, board_shape);

        let meta_shape = metadata_ndarray.shape().to_vec();
        let meta_vec = metadata_ndarray.into_raw_vec();
        let meta_data = TensorData::new(meta_vec, meta_shape);

        let device = Default::default();
        let input1 = Tensor::<B, 4>::from_data(board_data, &device);
        let input2 = Tensor::<B, 2>::from_data(meta_data, &device);

        let model = model.lock().expect("mutex poisoned");
        let output = model.forward(input1, input2);

        let best_idx = output.argmax(1).into_scalar() as usize;
        let moves = ["up", "right", "down", "left"];

        (moves[best_idx].to_string(), "🐍".to_string())
    })
    .await
    .expect("Blocking task failed");

    Json(MoveResponse {
        r#move: best_move,
        shout,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = burn_ndarray::NdArrayDevice::Cpu;

    println!("Loading Burn model...");
    let model: Model<B> = Model::from_file("simple_cnn_opset16", &device);
    let model = Arc::new(Mutex::new(model));

    let app = Router::new()
        .route("/", get(info))
        .route("/move", post(handle_move))
        .with_state(model);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    Ok(())
}
