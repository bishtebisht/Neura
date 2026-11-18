import { ENV } from "./env";

export type MLTeachingStrategy =
  | "visual"
  | "textual"
  | "step-by-step"
  | "example-based"
  | "socratic";

export type PredictStrategyInput = {
  previousAccuracy: number;
  responseTime: number;
  attempts: number;
  hintsRequested: number;
  topicMastery: number;
  previousStrategy: MLTeachingStrategy;
};

export type PredictStrategyResult = {
  nextStrategy: MLTeachingStrategy;
  confidence: number;
};

const REQUEST_TIMEOUT_MS = 3000;

/**
 * Calls the Python FastAPI strategy-prediction service. Returns null (rather
 * than throwing) on any failure -- network error, timeout, non-2xx, or a
 * malformed body -- so callers can fall back to the student's last known
 * strategy instead of breaking the lesson (see server/routers.ts
 * adaptive.predictNextStrategy).
 */
export async function predictNextStrategy(
  input: PredictStrategyInput
): Promise<PredictStrategyResult | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${ENV.mlServiceUrl.replace(/\/+$/, "")}/predict/strategy`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        previous_accuracy: input.previousAccuracy,
        response_time: input.responseTime,
        attempts: input.attempts,
        hints_requested: input.hintsRequested,
        topic_mastery: input.topicMastery,
        previous_strategy: input.previousStrategy,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(`[ML] /predict/strategy returned ${response.status}`);
      return null;
    }

    const body = (await response.json()) as { next_strategy?: string; confidence?: number };
    if (!body.next_strategy || typeof body.confidence !== "number") {
      console.warn("[ML] /predict/strategy returned an unexpected body:", body);
      return null;
    }

    return { nextStrategy: body.next_strategy as MLTeachingStrategy, confidence: body.confidence };
  } catch (error: any) {
    console.warn("[ML] /predict/strategy request failed, falling back:", error?.message);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
