import { describe, expect, it, vi, beforeEach } from "vitest";

const predictNextStrategyMock = vi.fn();
vi.mock("./_core/mlClient", () => ({
  predictNextStrategy: (...args: unknown[]) => predictNextStrategyMock(...args),
}));

import { appRouter } from "./routers";
import { getLatestLessonPerformance, recordStrategyObservationsBatch } from "./db";

function makeCaller(userId: number) {
  return appRouter.createCaller({
    user: { id: userId, openId: `test_loop_user_${userId}`, role: "user" },
    req: {} as any,
    res: { clearCookie: () => {} } as any,
  });
}

describe("Adaptive ML loop (adaptive.predictNextStrategy / getInitialStrategy)", () => {
  beforeEach(() => {
    predictNextStrategyMock.mockReset();
  });

  it("uses the ML service's prediction and persists the interaction when the service is reachable", async () => {
    predictNextStrategyMock.mockResolvedValue({ nextStrategy: "example-based", confidence: 0.81 });
    const caller = makeCaller(201);

    const result = await caller.adaptive.predictNextStrategy({
      topic: "Fractions",
      previousStrategy: "visual",
      accuracy: 0.72,
      responseTime: 15,
      attempts: 2,
      hintsRequested: 1,
      topicMastery: 0.68,
    });

    expect(result).toEqual({ nextStrategy: "example-based", confidence: 0.81, source: "ml" });

    const stored = await getLatestLessonPerformance(201, "Fractions");
    expect(stored?.strategyUsed).toBe("visual");
    expect(stored?.predictedNextStrategy).toBe("example-based");
    expect(stored?.predictedConfidence).toBe(0.81);
  });

  it("falls back to the previously-used strategy without crashing when the ML service is unreachable", async () => {
    predictNextStrategyMock.mockResolvedValue(null);
    const caller = makeCaller(202);

    const result = await caller.adaptive.predictNextStrategy({
      topic: "Fractions",
      previousStrategy: "step-by-step",
      accuracy: 0.5,
      responseTime: 20,
      attempts: 3,
      hintsRequested: 2,
      topicMastery: 0.4,
    });

    expect(result).toEqual({ nextStrategy: "step-by-step", confidence: null, source: "fallback" });

    const stored = await getLatestLessonPerformance(202, "Fractions");
    expect(stored?.strategyUsed).toBe("step-by-step");
    expect(stored?.predictedNextStrategy).toBeNull();
  });

  it.each([
    ["accuracy above 1", { accuracy: 1.5 }],
    ["negative response time", { responseTime: -5 }],
    ["zero attempts", { attempts: 0 }],
    ["negative hints", { hintsRequested: -1 }],
    ["mastery above 1", { topicMastery: 1.2 }],
    ["invalid strategy", { previousStrategy: "guessing" }],
  ])("rejects invalid input: %s", async (_label, overrides) => {
    const caller = makeCaller(203);
    const base = {
      topic: "Fractions",
      previousStrategy: "visual" as const,
      accuracy: 0.7,
      responseTime: 15,
      attempts: 2,
      hintsRequested: 1,
      topicMastery: 0.6,
    };

    await expect(
      caller.adaptive.predictNextStrategy({ ...base, ...overrides } as any)
    ).rejects.toThrow();
  });

  it("rejects a request missing required fields", async () => {
    const caller = makeCaller(204);
    await expect(
      caller.adaptive.predictNextStrategy({ topic: "Fractions" } as any)
    ).rejects.toThrow();
  });

  it("getInitialStrategy prefers the last ML-loop decision over the onboarding profile", async () => {
    predictNextStrategyMock.mockResolvedValue({ nextStrategy: "socratic", confidence: 0.9 });
    const caller = makeCaller(205);

    await recordStrategyObservationsBatch(205, "Fractions", [
      { concept: "c1", strategy: "visual", correct: 1, timeTaken: 10, attemptCount: 1, hintUsed: 0 },
    ]);

    const beforeLoop = await caller.adaptive.getInitialStrategy({ topic: "Fractions" });
    expect(beforeLoop.source).toBe("onboarding");

    await caller.adaptive.predictNextStrategy({
      topic: "Fractions",
      previousStrategy: "visual",
      accuracy: 0.8,
      responseTime: 10,
      attempts: 1,
      hintsRequested: 0,
      topicMastery: 0.7,
    });

    const afterLoop = await caller.adaptive.getInitialStrategy({ topic: "Fractions" });
    expect(afterLoop).toEqual({ strategy: "socratic", topicMastery: 0.7, source: "ml-loop" });
  });

  it("getInitialStrategy falls back cross-topic when this topic has no history, and returns null when the student has none at all", async () => {
    const freshUserCaller = makeCaller(9999);
    const noHistory = await freshUserCaller.adaptive.getInitialStrategy({ topic: "Geometry" });
    expect(noHistory).toEqual({ strategy: null, topicMastery: null, source: "none" });

    await recordStrategyObservationsBatch(206, "Photosynthesis", [
      { concept: "c1", strategy: "textual", correct: 1, timeTaken: 8, attemptCount: 1, hintUsed: 0 },
    ]);
    const otherUserCaller = makeCaller(206);
    const crossTopic = await otherUserCaller.adaptive.getInitialStrategy({ topic: "Geometry" });
    expect(crossTopic.source).toBe("cross-topic");
  });
});
