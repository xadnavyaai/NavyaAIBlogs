package main

import (
	"encoding/json"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	embedDim   = 384
	serverPort = ":8000"
)

// --- Request/Response types (match Python API for bench.py) ---

type IngestDocument struct {
	ID       *string           `json:"id,omitempty"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type IngestRequest struct {
	Documents []IngestDocument `json:"documents"`
}

type JobResult struct {
	JobID              string   `json:"job_id"`
	Status             string   `json:"status"`
	TotalDocuments     int      `json:"total_documents"`
	ProcessedDocuments int      `json:"processed_documents"`
	Error              *string  `json:"error,omitempty"`
	StartedAt          float64  `json:"started_at"`
	FinishedAt         *float64 `json:"finished_at,omitempty"`
}

type SearchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type SearchResult struct {
	ID       string            `json:"id"`
	Score    float64           `json:"score"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
	TookMs  float64        `json:"took_ms"`
}

type HealthResponse struct {
	Status        string `json:"status"`
	ModelLoaded   bool   `json:"model_loaded"`
	DocumentCount int    `json:"document_count"`
}

// --- Deterministic "embedding" (no ML deps; for benchmark parity) ---

func embed(text string) [embedDim]float64 {
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()
	var out [embedDim]float64
	for i := 0; i < embedDim; i++ {
		// deterministic pseudo-random in [0,1]
		seed = seed*6364136223846793005 + 1442695040888963407
		out[i] = float64(seed>>32) / (1 << 32)
	}
	// normalize to unit vector
	var norm float64
	for i := range out {
		norm += out[i] * out[i]
	}
	norm = math.Sqrt(norm)
	if norm < 1e-12 {
		norm = 1
	}
	for i := range out {
		out[i] /= norm
	}
	return out
}

func cosineSimilarity(a, b [embedDim]float64) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// --- In-memory index ---

type docEntry struct {
	id   string
	vec  [embedDim]float64
	text string
	meta map[string]string
}

type index struct {
	mu    sync.RWMutex
	docs  []docEntry
	count int
}

func (idx *index) documentCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.count
}

func (idx *index) ingest(docs []IngestDocument) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, d := range docs {
		id := uuid.New().String()
		if d.ID != nil && *d.ID != "" {
			id = *d.ID
		}
		meta := d.Metadata
		if meta == nil {
			meta = make(map[string]string)
		}
		idx.docs = append(idx.docs, docEntry{
			id:   id,
			vec:  embed(d.Text),
			text: d.Text,
			meta: meta,
		})
	}
	idx.count = len(idx.docs)
}

func (idx *index) search(query string, topK int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.docs) == 0 {
		return nil
	}
	q := embed(query)
	type scored struct {
		i     int
		score float64
	}
	scores := make([]scored, len(idx.docs))
	for i := range idx.docs {
		scores[i] = scored{i: i, score: cosineSimilarity(idx.docs[i].vec, q)}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if topK > len(scores) {
		topK = len(scores)
	}
	out := make([]SearchResult, topK)
	for k := 0; k < topK; k++ {
		e := idx.docs[scores[k].i]
		out[k] = SearchResult{ID: e.id, Score: scores[k].score, Text: e.text, Metadata: e.meta}
	}
	return out
}

// --- Jobs ---

var (
	vindex *index
	jobsMu sync.RWMutex
	jobs   map[string]*JobResult
)

func init() {
	vindex = &index{}
	jobs = make(map[string]*JobResult)
}

// --- Handlers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	jobID := uuid.New().String()
	now := float64(time.Now().UnixNano()) / 1e9
	job := &JobResult{
		JobID:          jobID,
		Status:         "pending",
		TotalDocuments: len(req.Documents),
		StartedAt:      now,
	}
	jobsMu.Lock()
	jobs[jobID] = job
	jobsMu.Unlock()

	go func() {
		job.Status = "running"
		job.ProcessedDocuments = len(req.Documents)
		vindex.ingest(req.Documents)
		job.Status = "completed"
		fin := float64(time.Now().UnixNano()) / 1e9
		job.FinishedAt = &fin
	}()

	writeJSON(w, http.StatusOK, job)
}

func handleJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	jobID := r.PathValue("job_id")
	if jobID == "" {
		http.Error(w, "job_id required", http.StatusBadRequest)
		return
	}
	jobsMu.RLock()
	job, ok := jobs[jobID]
	jobsMu.RUnlock()
	if !ok {
		http.Error(w, "unknown job_id", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}
	start := time.Now()
	results := vindex.search(req.Query, req.TopK)
	tookMs := time.Since(start).Seconds() * 1000
	writeJSON(w, http.StatusOK, SearchResponse{Results: results, TookMs: tookMs})
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	n := vindex.documentCount()
	writeJSON(w, http.StatusOK, HealthResponse{Status: "ok", ModelLoaded: true, DocumentCount: n})
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/ingest", handleIngest)
	mux.HandleFunc("GET /v1/jobs/{job_id}", handleJob)
	mux.HandleFunc("POST /v1/search", handleSearch)
	mux.HandleFunc("GET /healthz", handleHealthz)

	log.Printf("embedding gateway listening on %s", serverPort)
	if err := http.ListenAndServe(serverPort, mux); err != nil {
		log.Fatal(err)
	}
}
