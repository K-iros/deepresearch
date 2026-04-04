const baseURL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ResearchRequest {
  topic: string;
  search_api?: string;
  run_id?: string;
  trace_id?: string;
  resume_from_sequence?: number;
  mode?: string;
  max_sources?: number;
  concurrency?: number;
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

export interface StreamOptions {
  signal?: AbortSignal;
}

export interface RunListItem {
  run_id: string;
  topic: string;
  friendly_name: string;
  status: string;
  progress: number;
  created_at: string;
  updated_at: string;
  completed: boolean;
}

export interface RunListResponse {
  items: RunListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface RuntimeOptions {
  mode: string;
  max_sources: number;
  concurrency: number;
  stage_duration_stats: Record<string, number>;
  model_profile: Record<string, string>;
}

export interface ResumeRequest {
  resume_from_sequence?: number;
  search_api?: string;
  trace_id?: string;
  mode?: string;
  max_sources?: number;
  concurrency?: number;
}

function buildQueryString(params: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    query.set(key, String(value));
  }
  const text = query.toString();
  return text ? `?${text}` : "";
}

async function streamFromEndpoint(
  endpointPath: string,
  payload: unknown,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  const response = await fetch(`${baseURL}${endpointPath}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload),
    signal: options.signal
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(errorText || `研究请求失败，状态码：${response.status}`);
  }

  const body = response.body;
  if (!body) {
    throw new Error("浏览器不支持流式响应，无法获取研究进度");
  }

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent) {
        const lines = rawEvent.split("\n");
        let dataPayload = "";
        let sequence: number | undefined;

        for (const line of lines) {
          if (line.startsWith("id:")) {
            const value = Number(line.slice(3).trim());
            if (Number.isFinite(value)) {
              sequence = value;
            }
          }
          if (line.startsWith("data:")) {
            dataPayload += line.slice(5).trim();
          }
        }

        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            if (sequence !== undefined && typeof event.sequence !== "number") {
              event.sequence = sequence;
            }

            onEvent(event);

            const eventType =
              typeof event.event_type === "string"
                ? event.event_type
                : typeof event.type === "string"
                ? event.type
                : "";

            if (eventType === "error" || eventType === "done") {
              return;
            }
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }

      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      if (buffer.trim()) {
        const rawEvent = buffer.trim();
        const lines = rawEvent.split("\n");
        let dataPayload = "";
        let sequence: number | undefined;

        for (const line of lines) {
          if (line.startsWith("id:")) {
            const value = Number(line.slice(3).trim());
            if (Number.isFinite(value)) {
              sequence = value;
            }
          }
          if (line.startsWith("data:")) {
            dataPayload += line.slice(5).trim();
          }
        }

        if (dataPayload) {
          try {
            const event = JSON.parse(dataPayload) as ResearchStreamEvent;
            if (sequence !== undefined && typeof event.sequence !== "number") {
              event.sequence = sequence;
            }
            onEvent(event);
          } catch (error) {
            console.error("解析流式事件失败：", error, dataPayload);
          }
        }
      }
      break;
    }
  }
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint("/research/stream", payload, onEvent, options);
}

export async function resumeResearchRunStream(
  runId: string,
  payload: ResumeRequest,
  onEvent: (event: ResearchStreamEvent) => void,
  options: StreamOptions = {}
): Promise<void> {
  return streamFromEndpoint(`/research/runs/${encodeURIComponent(runId)}/resume`, payload, onEvent, options);
}

export async function listResearchRuns(params: {
  status?: string;
  keyword?: string;
  page?: number;
  page_size?: number;
}): Promise<RunListResponse> {
  const query = buildQueryString({
    status: params.status,
    keyword: params.keyword,
    page: params.page,
    page_size: params.page_size
  });
  const response = await fetch(`${baseURL}/research/runs${query}`, {
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `获取历史运行列表失败，状态码：${response.status}`);
  }

  return (await response.json()) as RunListResponse;
}

export async function getResearchRunDetail(
  runId: string,
  latestEventsLimit = 50
): Promise<Record<string, unknown>> {
  const query = buildQueryString({ latest_events_limit: latestEventsLimit });
  const response = await fetch(`${baseURL}/research/runs/${encodeURIComponent(runId)}${query}`, {
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `获取历史运行详情失败，状态码：${response.status}`);
  }

  return (await response.json()) as Record<string, unknown>;
}

export async function getRuntimeOptions(): Promise<RuntimeOptions> {
  const response = await fetch(`${baseURL}/research/runtime/options`, {
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `获取运行时配置失败，状态码：${response.status}`);
  }

  return (await response.json()) as RuntimeOptions;
}

export async function deleteResearchRun(runId: string): Promise<{ ok: boolean; run_id: string }> {
  const response = await fetch(`${baseURL}/research/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `删除历史记录失败，状态码：${response.status}`);
  }

  return (await response.json()) as { ok: boolean; run_id: string };
}
