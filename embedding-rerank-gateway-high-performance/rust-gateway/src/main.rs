//! Super gateway: one binary, three modes.
//! - ONNX: same model as Python (apples-to-apples).
//! - Remote: thin layer, calls EMBEDDING_API_URL (replaces Go thin use case).
//! - Pseudo: deterministic vectors for dev/bench (same as Go's no-ML mode).

use axum::{
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationStrategy};
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

const EMBED_DIM: usize = 384;
const MAX_LENGTH: usize = 128;
const SERVER_PORT: u16 = 8000;

// --- Request/Response types (match Python API for bench.py) ---

#[derive(Debug, Deserialize)]
struct IngestDocument {
    id: Option<String>,
    text: String,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IngestRequest {
    documents: Vec<IngestDocument>,
}

#[derive(Debug, Clone, Serialize)]
struct JobResult {
    job_id: String,
    status: String,
    total_documents: usize,
    processed_documents: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    started_at: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    finished_at: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
struct SearchResult {
    id: String,
    score: f64,
    text: String,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    took_ms: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
    document_count: usize,
}

// --- Embed-only service (Architecture C: embedding service) ---

#[derive(Debug, Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

async fn embed_handler(
    axum::extract::State(embedder): axum::extract::State<Arc<OnnxEmbedder>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    let arr = embedder.encode(&req.texts).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let embeddings: Vec<Vec<f32>> = (0..arr.nrows()).map(|i| arr.row(i).iter().copied().collect()).collect();
    Ok(Json(EmbedResponse { embeddings }))
}

// --- Embedding backends (trait + three implementations) ---

type EmbedResult = Result<Array2<f32>, Box<dyn std::error::Error + Send + Sync>>;

trait EmbeddingBackend: Send + Sync {
    fn encode(&self, texts: &[String]) -> EmbedResult;
    fn model_loaded(&self) -> bool;
}

// 1) ONNX: real model, same as Python (apples-to-apples).
struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl EmbeddingBackend for OnnxEmbedder {
    fn model_loaded(&self) -> bool {
        true
    }
    fn encode(&self, texts: &[String]) -> EmbedResult {
        onnx_encode_impl(&mut *self.session.lock().map_err(|e| format!("lock: {}", e))?, &self.tokenizer, texts)
    }
}

impl OnnxEmbedder {
    fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("tokenizer: {}", e))?;
        tokenizer.with_truncation(None); // truncation handled by manual seq_len cap below
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_LENGTH),
            direction: PaddingDirection::Right,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
            pad_to_multiple_of: None,
        }));
        Ok(Self { session: Mutex::new(session), tokenizer })
    }

}

fn onnx_encode_impl(session: &mut Session, tokenizer: &Tokenizer, texts: &[String]) -> EmbedResult {
    if texts.is_empty() {
        return Ok(Array2::zeros((0, EMBED_DIM)));
    }
    let encodings: Vec<_> = texts
            .iter()
            .map(|s| tokenizer.encode(s.as_str(), true).map_err(|e| format!("encode: {}", e)))
            .collect::<Result<Vec<_>, _>>()?;

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_LENGTH);

        let mut input_ids = vec![0i64; batch_size * seq_len];
        let mut attention_mask = vec![0i64; batch_size * seq_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let am = enc.get_attention_mask();
            for (j, (&id, &mask)) in ids.iter().zip(am.iter()).take(seq_len).enumerate() {
                input_ids[i * seq_len + j] = id as i64;
                attention_mask[i * seq_len + j] = mask as i64;
            }
            for j in ids.len().min(seq_len)..seq_len {
                input_ids[i * seq_len + j] = 0;
                attention_mask[i * seq_len + j] = 0;
            }
        }

        let input_ids_arr = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| format!("input_ids shape: {}", e))?;
        let attention_mask_arr = ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| format!("attention_mask shape: {}", e))?;

        let input_ids_tensor = Tensor::from_array(input_ids_arr)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_arr)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;

        // BERT feature-extraction export: output "last_hidden_state" [batch, seq, 384]
        let view = outputs["last_hidden_state"].try_extract_array::<f32>()?;
        let shape = view.shape();

        let embeddings: Array2<f32> = if shape.len() == 2 {
            let (b, d) = (shape[0], shape[1]);
            let mut arr = Array2::zeros((b, d));
            for i in 0..b {
                for j in 0..d {
                    arr[[i, j]] = view[[i, j]];
                }
            }
            arr
        } else if shape.len() == 3 {
            let (b, s, d) = (shape[0], shape[1], shape[2]);
            let mut pooled = Array2::zeros((b, d));
            for i in 0..b {
                let mut sum = 0f32;
                for j in 0..s {
                    for k in 0..d {
                        pooled[[i, k]] += view[[i, j, k]];
                    }
                    sum += 1.0;
                }
                if sum > 0.0 {
                    for k in 0..d {
                        pooled[[i, k]] /= sum;
                    }
                }
            }
            pooled
        } else {
            return Err("unexpected output shape".into());
        };

        let mut out = Array2::zeros((batch_size, EMBED_DIM));
        for i in 0..batch_size {
            let mut norm = 0f32;
            for j in 0..EMBED_DIM {
                let v = embeddings[[i, j]];
                norm += v * v;
            }
            norm = norm.sqrt().max(1e-12);
            for j in 0..EMBED_DIM {
                out[[i, j]] = embeddings[[i, j]] / norm;
            }
        }
        Ok(out)
}

// 2) Pseudo: deterministic 384-d vectors (same idea as Go gateway, no ML).
struct PseudoEmbedder;

impl EmbeddingBackend for PseudoEmbedder {
    fn model_loaded(&self) -> bool {
        false
    }
    fn encode(&self, texts: &[String]) -> EmbedResult {
        let mut out = Array2::zeros((texts.len(), EMBED_DIM));
        for (i, text) in texts.iter().enumerate() {
            let vec = pseudo_embed(text);
            for j in 0..EMBED_DIM {
                out[[i, j]] = vec[j];
            }
        }
        Ok(out)
    }
}

fn pseudo_embed(text: &str) -> [f32; EMBED_DIM] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    let mut out = [0f32; EMBED_DIM];
    let mut s = seed;
    for i in 0..EMBED_DIM {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out[i] = (s >> 32) as f32 / (1u64 << 32) as f32;
    }
    let mut norm = 0f32;
    for x in &out {
        norm += x * x;
    }
    let norm = norm.sqrt().max(1e-12);
    for x in &mut out {
        *x /= norm;
    }
    out
}

// 3) Remote: thin layer, call external embedding API (replaces Go thin use case).
struct RemoteEmbedder {
    client: reqwest::Client,
    url: String,
}

impl EmbeddingBackend for RemoteEmbedder {
    fn model_loaded(&self) -> bool {
        true
    }
    fn encode(&self, texts: &[String]) -> EmbedResult {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, EMBED_DIM)));
        }
        let url = self.url.clone();
        let texts = texts.to_vec();
        let client = self.client.clone();
        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                client
                    .post(&url)
                    .json(&serde_json::json!({ "texts": texts }))
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<RemoteEmbeddingResponse>()
                    .await
            })
        })?;
        let n = res.embeddings.len();
        let mut arr = Array2::zeros((n, EMBED_DIM));
        for (i, row) in res.embeddings.iter().enumerate() {
            for (j, &v) in row.iter().take(EMBED_DIM).enumerate() {
                arr[[i, j]] = v;
            }
            let mut norm = 0f32;
            for j in 0..EMBED_DIM {
                norm += arr[[i, j]] * arr[[i, j]];
            }
            let norm = norm.sqrt().max(1e-12);
            for j in 0..EMBED_DIM {
                arr[[i, j]] /= norm;
            }
        }
        Ok(arr)
    }
}

#[derive(Deserialize)]
struct RemoteEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

// --- In-memory index ---

struct DocEntry {
    id: String,
    vec: Vec<f32>,
    text: String,
    metadata: HashMap<String, String>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
    }
    dot
}

struct Index {
    docs: Vec<DocEntry>,
    embedder: Arc<dyn EmbeddingBackend + Send + Sync>,
}

impl Index {
    fn document_count(&self) -> usize {
        self.docs.len()
    }

    fn ingest(&mut self, docs: Vec<IngestDocument>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let vectors = self.embedder.encode(&texts)?;
        for (i, d) in docs.into_iter().enumerate() {
            let id = d.id.filter(|s| !s.is_empty()).unwrap_or_else(|| Uuid::new_v4().to_string());
            let vec: Vec<f32> = vectors.row(i).iter().copied().collect();
            self.docs.push(DocEntry {
                id,
                vec,
                text: texts[i].clone(),
                metadata: d.metadata,
            });
        }
        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
        if self.docs.is_empty() {
            return Ok(vec![]);
        }
        let q_vec = self.embedder.encode(&[query.to_string()])?;
        let q: Vec<f32> = q_vec.row(0).iter().copied().collect();

        let mut scored: Vec<(usize, f64)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, cosine_similarity(&d.vec, &q)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(scored.len());
        let results: Vec<SearchResult> = scored
            .into_iter()
            .take(k)
            .map(|(i, score)| {
                let d = &self.docs[i];
                SearchResult {
                    id: d.id.clone(),
                    score,
                    text: d.text.clone(),
                    metadata: d.metadata.clone(),
                }
            })
            .collect();
        Ok(results)
    }
}

// --- App state ---

struct AppState {
    index: RwLock<Index>,
    jobs: RwLock<HashMap<String, JobResult>>,
    model_loaded: bool,
}

// --- Handlers ---

async fn ingest(axum::extract::State(state): axum::extract::State<Arc<AppState>>, Json(req): Json<IngestRequest>) -> impl IntoResponse {
    let job_id = Uuid::new_v4().to_string();
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let total = req.documents.len();
    let job = JobResult {
        job_id: job_id.clone(),
        status: "pending".to_string(),
        total_documents: total,
        processed_documents: 0,
        error: None,
        started_at,
        finished_at: None,
    };
    state.jobs.write().await.insert(job_id.clone(), job.clone());

    let state_clone = state.clone();
    let docs = req.documents;
    let job_id_inner = job_id.clone();
    tokio::spawn(async move {
        let mut jobs = state_clone.jobs.write().await;
        if let Some(j) = jobs.get_mut(&job_id_inner) {
            j.status = "running".to_string();
        }
        drop(jobs);
        {
            let mut index = state_clone.index.write().await;
            if let Err(e) = index.ingest(docs) {
                let mut jobs = state_clone.jobs.write().await;
                if let Some(j) = jobs.get_mut(&job_id_inner) {
                    j.status = "failed".to_string();
                    j.error = Some(format!("{}", e));
                }
            } else {
                let mut jobs = state_clone.jobs.write().await;
                if let Some(j) = jobs.get_mut(&job_id_inner) {
                    j.status = "completed".to_string();
                    j.processed_documents = total;
                }
            }
        }
        let mut jobs = state_clone.jobs.write().await;
        if let Some(j) = jobs.get_mut(&job_id_inner) {
            j.finished_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            );
        }
    });

    Json(JobResult {
        job_id,
        status: "pending".to_string(),
        total_documents: total,
        processed_documents: 0,
        error: None,
        started_at,
        finished_at: None,
    })
}

async fn get_job(
    Path(job_id): Path<String>,
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<Json<JobResult>, (StatusCode, String)> {
    let jobs = state.jobs.read().await;
    jobs.get(&job_id)
        .cloned()
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown job_id".to_string()))
}

async fn search(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let index = state.index.read().await;
    let top_k = if req.top_k <= 0 { 5 } else { req.top_k };
    let results = index.search(&req.query, top_k).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SearchResponse { results, took_ms }))
}

async fn healthz(axum::extract::State(state): axum::extract::State<Arc<AppState>>) -> Json<HealthResponse> {
    let count = state.index.read().await.document_count();
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: state.model_loaded,
        document_count: count,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    // Architecture C: embedding-only service (POST /embed only). Same binary, different mode.
    if std::env::var("SERVE_EMBED_ONLY").as_deref() == Ok("true") {
        let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "model".to_string());
        let model_path = format!("{}/model.onnx", model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        if !std::path::Path::new(&model_path).exists() {
            return Err(format!(
                "SERVE_EMBED_ONLY: ONNX model not found at {}. Set MODEL_DIR and run export_onnx.py.",
                model_path
            )
            .into());
        }
        let embedder = Arc::new(OnnxEmbedder::new(&model_path, &tokenizer_path)?);
        let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        info!("Embed-only service (Architecture C) listening on {}", addr);
        let app = Router::new()
            .route("/embed", post(embed_handler))
            .with_state(embedder);
        axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
        return Ok(());
    }

    let (embedder, model_loaded) = match (
        std::env::var("MODEL_DIR").ok().filter(|d| !d.is_empty()),
        std::env::var("EMBEDDING_API_URL").ok().filter(|u| !u.is_empty()),
    ) {
        (Some(model_dir), _) => {
            let model_path = format!("{}/model.onnx", model_dir);
            let tokenizer_path = format!("{}/tokenizer.json", model_dir);
            if !std::path::Path::new(&model_path).exists() {
                return Err(format!(
                    "ONNX model not found at {}. Run: pip install 'optimum[onnx]' && python scripts/export_onnx.py -o {}",
                    model_path, model_dir
                )
                .into());
            }
            let e = Arc::new(OnnxEmbedder::new(&model_path, &tokenizer_path)?) as Arc<dyn EmbeddingBackend + Send + Sync>;
            info!("Super gateway mode: ONNX (model from {})", model_dir);
            (e, true)
        }
        (_, Some(url)) => {
            let e = Arc::new(RemoteEmbedder {
                client: reqwest::Client::new(),
                url,
            }) as Arc<dyn EmbeddingBackend + Send + Sync>;
            info!("Super gateway mode: Remote (EMBEDDING_API_URL)");
            (e, true)
        }
        (None, None) => {
            let e = Arc::new(PseudoEmbedder) as Arc<dyn EmbeddingBackend + Send + Sync>;
            info!("Super gateway mode: Pseudo (no model; set MODEL_DIR or EMBEDDING_API_URL for real embeddings)");
            (e, false)
        }
    };

    let state = Arc::new(AppState {
        index: RwLock::new(Index {
            docs: vec![],
            embedder,
        }),
        jobs: RwLock::new(HashMap::new()),
        model_loaded,
    });

    let app = Router::new()
        .route("/v1/ingest", post(ingest))
        .route("/v1/jobs/:job_id", get(get_job))
        .route("/v1/search", post(search))
        .route("/healthz", get(healthz))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], SERVER_PORT));
    info!("embedding gateway listening on {}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
