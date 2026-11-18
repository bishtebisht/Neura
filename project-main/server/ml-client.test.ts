import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { predictNextStrategy } from "./_core/mlClient";

describe("mlClient.predictNextStrategy failure & retry observability", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("returns { ok: true, data } on successful response", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ next_strategy: "example-based", confidence: 0.88 }),
    });

    const res = await predictNextStrategy({
      topicMastery: 0.5,
      previousAccuracy: 0.7,
      responseTime: 20,
      attempts: 1,
      hintsRequested: 0,
      previousStrategy: "visual",
    });

    expect(res).toEqual({
      ok: true,
      data: {
        nextStrategy: "example-based",
        confidence: 0.88,
      },
    });
  });

  it("returns { ok: false, reason: 'http_error', status: 404 } on 4xx error without retrying", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
    });
    global.fetch = fetchMock;

    const res = await predictNextStrategy({
      topicMastery: 0.5,
      previousAccuracy: 0.7,
      responseTime: 20,
      attempts: 1,
      hintsRequested: 0,
      previousStrategy: "visual",
    });

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("http_error");
      expect(res.status).toBe(404);
    }
    expect(fetchMock).toHaveBeenCalledTimes(1); // No pointless retry on 4xx
  });

  it("returns { ok: false, reason: 'invalid_response' } when payload misses required fields", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ invalid: true }),
    });

    const res = await predictNextStrategy({
      topicMastery: 0.5,
      previousAccuracy: 0.7,
      responseTime: 20,
      attempts: 1,
      hintsRequested: 0,
      previousStrategy: "visual",
    });

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("invalid_response");
    }
  });

  it("retries once on transient network failure before returning { ok: false, reason: 'network' }", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("fetch failed: ECONNREFUSED"));
    global.fetch = fetchMock;

    const res = await predictNextStrategy({
      topicMastery: 0.5,
      previousAccuracy: 0.7,
      responseTime: 20,
      attempts: 1,
      hintsRequested: 0,
      previousStrategy: "visual",
    });

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("network");
    }
    expect(fetchMock).toHaveBeenCalledTimes(2); // 1 initial + 1 retry
  });

  it("retries once on timeout error before returning { ok: false, reason: 'timeout' }", async () => {
    const abortError = new Error("The operation was aborted");
    abortError.name = "AbortError";
    const fetchMock = vi.fn().mockRejectedValue(abortError);
    global.fetch = fetchMock;

    const res = await predictNextStrategy({
      topicMastery: 0.5,
      previousAccuracy: 0.7,
      responseTime: 20,
      attempts: 1,
      hintsRequested: 0,
      previousStrategy: "visual",
    });

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("timeout");
    }
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
