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

export type PredictStrategyFailureReason =
  | "timeout"
  | "network"
  | "http_error"
  | "invalid_response";

export type PredictStrategyResponse =
  | { ok: true; data: PredictStrategyResult }
  | {
      ok: false;
      reason: PredictStrategyFailureReason;
      status?: number;
      error?: string;
    };

const REQUEST_TIMEOUT_MS = 3000;
const RETRY_BACKOFF_MS = 300;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Calls the Python FastAPI strategy-prediction service with timeout, typed failure
 * discrimination, and single retry on transient network/timeout errors.
 */
export async function predictNextStrategy(
  input: PredictStrategyInput
): Promise<PredictStrategyResponse> {
  const url = `${ENV.mlServiceUrl.replace(/\/+$/, "")}/predict/strategy`;
  const bodyPayload = JSON.stringify({
    previous_accuracy: input.previousAccuracy,
    response_time: input.responseTime,
    attempts: input.attempts,
    hints_requested: input.hintsRequested,
    topic_mastery: input.topicMastery,
    previous_strategy: input.previousStrategy,
  });

  let lastFailure: PredictStrategyResponse = { ok: false, reason: "network" };

  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0) {
      await sleep(RETRY_BACKOFF_MS);
    }

    const controller = new AbortController();
    let isTimeout = false;
    const timeout = setTimeout(() => {
      isTimeout = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: bodyPayload,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        // 4xx or 5xx: HTTP error, do not retry
        return {
          ok: false,
          reason: "http_error",
          status: response.status,
          error: `HTTP ${response.status} ${response.statusText}`,
        };
      }

      let body: any;
      try {
        body = await response.json();
      } catch (jsonErr: any) {
        return {
          ok: false,
          reason: "invalid_response",
          error: "Failed to parse JSON response",
        };
      }

      if (!body?.next_strategy || typeof body?.confidence !== "number") {
        return {
          ok: false,
          reason: "invalid_response",
          error: "Missing required fields in response payload",
        };
      }

      return {
        ok: true,
        data: {
          nextStrategy: body.next_strategy as MLTeachingStrategy,
          confidence: body.confidence,
        },
      };
    } catch (error: any) {
      clearTimeout(timeout);
      const isAbort = error?.name === "AbortError" || isTimeout;
      lastFailure = {
        ok: false,
        reason: isAbort ? "timeout" : "network",
        error: error?.message || String(error),
      };
      // Loop continues for attempt 1 -> attempt 2 retry on network/timeout only
    }
  }

  return lastFailure;
}
