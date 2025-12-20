use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn info() -> Json<InfoResponse> {
    Json(InfoResponse {
        apiversion: "1".into(),
        color: "#5a7576".into(),
        head: "cute-dragon".into(),
    })
}

async fn handle_move(
    State(session): State<Arc<Mutex<Session>>>,
    Json(req): Json<GameMoveRequest>,
) -> Json<MoveResponse> {
    let (board, metadata) = preprocess(&req);

    let board_input = Value::from_array(board).expect("failed to create board tensor");
    let metadata_input = Value::from_array(metadata).expect("failed to create metadata tensor");
    let mut session = session.lock().await;
    let outputs = session
        .run(ort::inputs!["board_input" => board_input, "metadata_input" => metadata_input])
        .expect("inference");

    let output = outputs["output"]
        .try_extract_tensor::<f32>()
        .expect("extract");

    let logits: Vec<f32> = output.1.iter().cloned().collect();
    let probs = softmax(&logits);

    let moves = ["up", "right", "down", "left"];
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("Logits: {:?}", logits);
    println!("Probs:  {:?}", probs);
    println!("Move:   {}", moves[best]);

    Json(MoveResponse {
        r#move: moves[best].into(),
        shout: "🐍🐍🐍🐍".into(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ort::init().commit()?;
    let session = Session::builder()?.commit_from_file("simple_cnn_prototype.onnx")?;
    let session = Arc::new(Mutex::new(session));

    let app = Router::new()
        .route("/", get(info))
        .route("/move", post(handle_move))
        .with_state(session);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    Ok(())
}
