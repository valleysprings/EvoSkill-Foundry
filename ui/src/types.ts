export type RuntimeInfo = {
  mode: string;
  primary_model: string;
  active_model: string;
  available_models: string[];
  api_base: string;
  temperature: number | string;
  max_tokens: number | string;
  timeout_s: number | string;
};

export type TaskSummary = {
  id: string;
  title: string;
  description: string;
  family: string;
  function_name: string;
  objective_label: string;
  objective_direction: string;
  generation_budget: number;
  candidate_budget: number;
};

export type CandidateMetrics = {
  objective: number | string;
  J: number | string;
  benchmark_ms?: number | null;
  speedup_vs_baseline?: number | string;
  passed_tests?: number | string;
  total_tests?: number | string;
  verifier_status?: string;
  status?: string;
};

export type Candidate = {
  candidate_id?: string;
  agent: string;
  label: string;
  strategy: string;
  rationale: string;
  candidate_summary: string;
  proposal_model?: string | null;
  verifier_status?: string;
  workspace_path?: string;
  source_code: string;
  metrics: CandidateMetrics;
};

export type MemoryFragment = {
  experience_id: string;
  prompt_fragment?: string;
  strategy_hypothesis?: string;
};

export type AddedExperience = {
  experience_id: string;
  generation: number;
  experience_outcome: string;
  verifier_status: string;
  delta_J: number | string;
  prompt_fragment: string;
  strategy_hypothesis: string;
  candidate_summary: string;
  proposal_model?: string | null;
};

export type Generation = {
  generation: number;
  winner_accepted: boolean;
  wrote_memory: boolean;
  delta_J: number | string;
  winner: Candidate;
  retrieved_memories: MemoryFragment[];
  candidates: Candidate[];
};

export type ObjectivePoint = {
  generation: number;
  objective: number | string;
  candidate_objective: number | string;
  J: number | string;
  candidate_J: number | string;
  accepted: boolean;
};

export type RunTask = {
  id: string;
  title: string;
  description: string;
  family: string;
  function_name: string;
  function_signature: string;
  objective_label: string;
  objective_direction: string;
  generation_budget: number;
  candidate_budget: number;
  source_type: string;
};

export type ArtifactManifest = {
  artifact_paths: {
    payload: string;
    trace: string;
    llm_trace_jsonl: string;
    memory_markdown: string;
    report_svg?: string;
    improvement_table_json?: string;
  };
};

export type ImprovementRow = {
  label: string;
  value: string;
};

export type Run = {
  run_mode: string;
  active_model: string;
  session_id?: string;
  generated_at?: string;
  task: RunTask;
  baseline: Candidate;
  winner: Candidate;
  delta_J: number | string;
  objective_curve: ObjectivePoint[];
  improvement_table?: ImprovementRow[];
  llm_traces: Array<Record<string, unknown>>;
  generations: Generation[];
  memory_before_count?: number | string;
  memory_after_count?: number | string;
  positive_experiences_added?: number | string;
  negative_experiences_added?: number | string;
  added_experiences?: AddedExperience[];
  memory_markdown: string;
  selection_reason: string;
  handoff_bundle?: {
    manifest: ArtifactManifest;
    manifest_path: string;
  };
};

export type PayloadSummary = {
  generated_at: string;
  run_mode: string;
  active_model: string;
  num_tasks: number;
  total_generations: number;
  initial_memory_count: number;
  memory_size_after_run: number;
  write_backs: number;
  source_repo: string;
  git_commit: string;
  upstream_target: string;
  flywheel: string[];
  proposal_engine: RuntimeInfo;
};

export type Payload = {
  summary: PayloadSummary;
  formulas: {
    J: string;
    objective: string;
    delta_J: string;
  };
  audit: {
    workspace_root: string;
    session_id?: string | null;
  };
  task_catalog: TaskSummary[];
  runs: Run[];
};

export type ErrorPayload = {
  terminal: boolean;
  error_type: string;
  error: string;
  model: string | null;
};

export type LiveEvent = {
  phase?: string;
  event_type?: string;
  generation?: number;
  candidate?: string;
  timestamp?: string;
  message?: string;
};

export type JobState = {
  status: "loading" | "running" | "failed" | "completed";
  task_id?: string | null;
  taskId?: string | null;
  terminal?: boolean;
  error_type?: string | null;
  error?: string | null;
  model?: string | null;
  events: LiveEvent[];
  payload?: Payload;
};
