// Typed client for the DRS backend. Base URL is configurable via NEXT_PUBLIC_API_BASE.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8100").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string, public requestId?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, `Backend unreachable at ${API_BASE}`);
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, msg, res.headers.get("X-Request-ID") ?? undefined);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// ---------------------------------------------------------------- types
export interface VideoMeta {
  path_name: string;
  container_format: string | null;
  codec: string | null;
  width: number;
  height: number;
  nominal_fps: number | null;
  frame_count_declared: number | null;
  duration_s: number | null;
  variable_frame_rate: boolean | null;
  pts_delta_ms_mean: number | null;
  has_audio: boolean;
  size_bytes: number | null;
  warnings: string[];
}
export interface Video {
  id: string;
  original_filename: string;
  size_bytes: number;
  meta: VideoMeta;
  created_at: string;
}
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export interface JobSummary {
  pipeline_version: string;
  detector: { model_id: string; cricket_specific: boolean; notes: string[] };
  frames_processed: number;
  frames_with_candidates: number;
  num_detections: number;
  num_tracks: number;
  primary_track: TrackQuality | null;
  trajectory_2d: { status: string; model: string; rms_px: number; loo_rms_px: number | null; change_point_time_s: number | null } | null;
  timings: Record<string, number>;
  throughput_fps: { detect: number | null; end_to_end: number };
  gpu_peak_mb: number | null;
  warnings: string[];
  validation_status: string;
  config: Record<string, unknown>;
}
export interface Job {
  id: string;
  video_id: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  config: Record<string, unknown>;
  summary: JobSummary | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}
export type PointKind = "observed" | "filtered" | "interpolated" | "predicted" | "manual";
export interface TrackPoint {
  frame_index: number;
  timestamp_ms: number;
  kind: PointKind;
  x: number;
  y: number;
  sigma_x: number | null;
  sigma_y: number | null;
  confidence: number | null;
  detection_source: string | null;
}
export interface TrackQuality {
  num_frames_spanned: number;
  num_observed: number;
  num_missing: number;
  observed_ratio: number;
  longest_gap: number;
  mean_innovation_px: number | null;
  mean_confidence: number | null;
  fragments_merged: number;
  reliable: boolean;
  reliability_issues: string[];
}
export interface BallTrack {
  track_id: number;
  is_primary: boolean;
  points: TrackPoint[];
  quality: TrackQuality;
  score: number;
}
export interface Detection {
  frame_index: number;
  timestamp_ms: number;
  bbox: [number, number, number, number];
  center: [number, number];
  confidence: number;
  model_id: string;
  source: string;
}
export interface FrameDetections {
  frame_index: number;
  timestamp_ms: number;
  timestamp_source: string;
  ball_detected: boolean;
  detections: Detection[];
}
export interface JobResults {
  job: Job;
  summary: JobSummary;
  tracking: { tracks: BallTrack[]; primary_track_id: number | null; num_frames: number; warnings: string[]; fragmentation_note: string | null };
  trajectory_2d: { status: string; model_name: string | null; residual_rms_px: number | null; reasons: string[] } | null;
  metadata: VideoMeta;
}
export interface Annotation {
  id: string;
  frame_index: number;
  action: "add" | "move" | "reject" | "mark_absent";
  label: string;
  data: { x?: number; y?: number; note?: string };
  created_at: string;
}
export interface CameraModel {
  camera_id: string;
  image_width: number;
  image_height: number;
  K: number[][];
  R: number[][];
  t: number[];
  calibration_type: string;
  validated: boolean;
  reprojection_rms_px: number | null;
  num_correspondences: number | null;
  assumptions: string[];
  warnings: string[];
}
export interface Calibration {
  id: string;
  video_id: string | null;
  camera: CameraModel;
  correspondences: { points?: { name?: string; image: [number, number] }[]; per_point_error_px?: number[] };
  created_at: string;
}
export interface AutoCalibration {
  method: "two_wicket_stumps" | "crease_registration" | "seeds_only";
  frame_index: number;
  points: { name: string; image: [number, number] }[];
  evidence: Record<string, number>;
  warnings: string[];
  focal_px: number | null;
  note: string;
}
export interface PointEstimate {
  mean: number[];
  std: number[];
  p05: number[];
  p95: number[];
  n_valid: number;
}
export interface RuleCheck {
  id: string;
  description: string;
  law_reference: string;
  status: "PASS" | "FAIL" | "UNKNOWN" | "NOT_APPLICABLE";
  probability: number | null;
  evidence: string;
}
export type Verdict = "PROJECTED_TO_HIT" | "PROJECTED_TO_MISS" | "INCONCLUSIVE" | "INSUFFICIENT_DATA";
export interface LbwResult {
  verdict: Verdict;
  p_hit: number | null;
  rules_summary: string;
  rule_checks: RuleCheck[];
  pitching_point: PointEstimate | null;
  impact_point: PointEstimate | null;
  impact_time_s: number | null;
  impact_time_source: string | null;
  wicket_projection: {
    crossing: PointEstimate | null;
    p_hit: number | null;
    p_centre_in_zone: number | null;
    p_below_ground: number;
    n_samples: number;
    projection_distance_m: number | null;
    nominal_crossing_xz: number[] | null;
  } | null;
  predicted_path: number[][];
  observed_path_fit: number[][];
  measurement_uncertainty: Record<string, unknown>;
  model_uncertainty: Record<string, unknown>;
  input_quality: Record<string, unknown>;
  reasons: string[];
  warnings: string[];
  reconstruction_mode: string | null;
  calibration_validated: boolean;
  engine_version: string;
  disclaimer: string;
  overlay: {
    observations_px?: number[][];
    fitted_path_px?: (number[] | null)[];
    predicted_path_px?: (number[] | null)[];
    wicket_outline_px?: (number[] | null)[];
    striker_crease_px?: (number[] | null)[];
    pitching_point_px?: number[] | null;
    impact_point_px?: number[] | null;
    stump_crossing_px?: number[] | null;
    stump_crossing_2sigma_px?: (number[] | null)[];
    notes?: string[];
  };
}
export interface LbwAnalysis {
  id: string;
  job_id: string;
  calibration_id: string;
  request: Record<string, unknown>;
  result: LbwResult;
  created_at: string;
}
export interface LbwFacts {
  batter_handedness: "right" | "left" | null;
  no_ball: boolean | null;
  ball_touched_bat_first: boolean | null;
  genuine_attempt_to_play: boolean | null;
  intercepted_by_person_not_bat_hand: boolean | null;
}
export interface SystemStatus {
  platform: string;
  python: string;
  torch: { available: boolean; version?: string; cuda_available?: boolean; device?: string; vram_total_gb?: number; error?: string };
  gpu: { name: string; driver: string; memory_total_mb: number; memory_used_mb: number; utilization_pct: number } | null;
  models: { file: string; size_mb: number; card: Record<string, unknown> | null }[];
  cricket_detector_available: boolean;
  effective_detector_mode: string;
}
export interface Stats {
  videos: number;
  jobs_by_status: Record<string, number>;
  lbw_analyses: number;
  jobs_with_primary_track: number;
  jobs_succeeded: number;
  detection_metrics: { file: string; summary: Record<string, unknown> } | null;
  trajectory_benchmark: { file: string; summary: Record<string, unknown> } | null;
}
export interface DatasetEntry {
  id: string;
  name: string;
  source: string;
  url: string;
  license: string;
  commercial_use: string;
  size: string;
  num_items: unknown;
  labels: string[];
  has_ball_annotations: boolean | string;
  has_trajectories_or_3d: boolean | string;
  multi_view: boolean | string;
  auth_required: boolean | string;
  intended_use: string;
  verification: string;
  status: string;
  last_error?: string | null;
  download_note?: string;
}

// ---------------------------------------------------------------- calls
export const api = {
  health: () => request<{ status: string }>("/api/health"),
  system: () => request<SystemStatus>("/api/system/status"),
  stats: () => request<Stats>("/api/stats"),
  datasets: () => request<{ datasets: DatasetEntry[]; notes: string; last_reviewed: string }>("/api/datasets"),
  evaluationReports: () => request<{ file: string; summary: Record<string, unknown> | null }[]>("/api/evaluation/reports"),
  videos: () => request<Video[]>("/api/videos"),
  video: (id: string) => request<Video>(`/api/videos/${id}`),
  uploadVideo: (file: File, onProgress?: (frac: number) => void) =>
    new Promise<Video>((resolve, reject) => {
      // XHR gives upload progress events, which fetch does not.
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/videos`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        let body: { detail?: unknown } = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Video);
        else reject(new ApiError(xhr.status, typeof body.detail === "string" ? body.detail : xhr.statusText));
      };
      xhr.onerror = () => reject(new ApiError(0, `Backend unreachable at ${API_BASE}`));
      const fd = new FormData();
      fd.append("file", file);
      xhr.send(fd);
    }),
  jobs: (videoId?: string) => request<Job[]>(`/api/analysis/jobs${videoId ? `?video_id=${videoId}` : ""}`),
  job: (id: string) => request<Job>(`/api/analysis/jobs/${id}`),
  createJob: (body: Record<string, unknown>) => request<Job>("/api/analysis/jobs", json(body)),
  cancelJob: (id: string) => request<Job>(`/api/analysis/jobs/${id}/cancel`, { method: "POST" }),
  results: (id: string) => request<JobResults>(`/api/analysis/jobs/${id}/results`),
  detections: (id: string) => request<FrameDetections[]>(`/api/analysis/jobs/${id}/detections`),
  annotations: (videoId: string) => request<Annotation[]>(`/api/videos/${videoId}/annotations`),
  addAnnotation: (videoId: string, body: Record<string, unknown>) => request<Annotation>(`/api/videos/${videoId}/annotations`, json(body)),
  deleteAnnotation: (videoId: string, id: string) =>
    request<void>(`/api/videos/${videoId}/annotations/${id}`, { method: "DELETE" }),
  landmarks: () => request<{ frame: string; landmarks: Record<string, [number, number, number]> }>("/api/calibration/landmarks"),
  calibrations: (videoId: string) => request<Calibration[]>(`/api/calibration?video_id=${videoId}`),
  createCalibration: (body: Record<string, unknown>) => request<Calibration>("/api/calibration", json(body)),
  autoCalibration: (body: { video_id: string; frame_index?: number }) => request<AutoCalibration>("/api/calibration/auto", json(body)),
  lbwList: (jobId: string) => request<LbwAnalysis[]>(`/api/lbw?job_id=${jobId}`),
  lbwAnalyze: (body: Record<string, unknown>) => request<LbwAnalysis>("/api/lbw/analyze", json(body)),
  contact: (jobId: string) => request<Record<string, unknown>>("/api/contact/analyze", json({ job_id: jobId })),
};

export const urls = {
  videoFile: (id: string) => `${API_BASE}/api/videos/${id}/file`,
  frame: (videoId: string, idx: number) => `${API_BASE}/api/videos/${videoId}/frame/${idx}`,
  artifact: (jobId: string, name: string) => `${API_BASE}/api/analysis/jobs/${jobId}/artifacts/${name}`,
  report: (jobId: string) => `${API_BASE}/api/analysis/jobs/${jobId}/report.pdf`,
  exportJson: (jobId: string) => `${API_BASE}/api/analysis/jobs/${jobId}/export.json`,
  docs: () => `${API_BASE}/docs`,
};
