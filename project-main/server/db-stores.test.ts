import { describe, expect, it, beforeEach } from "vitest";
import {
  CappedArray,
  CappedMap,
  calculateStrategyScore,
  getInMemoryRecordCounts,
  getInMemoryStoresForTesting,
  recordStrategyObservationsBatch,
  resetInMemoryStoresForTesting,
} from "./db";

describe("In-memory DB fallback stores & observation recording", () => {
  beforeEach(() => {
    resetInMemoryStoresForTesting();
  });

  it("CappedMap evicts oldest entry when reaching max size", () => {
    const map = new CappedMap<string, number>(3);
    map.set("a", 1);
    map.set("b", 2);
    map.set("c", 3);
    expect(map.size).toBe(3);
    expect(Array.from(map.keys())).toEqual(["a", "b", "c"]);

    // Insert 4th element -> "a" should be evicted
    map.set("d", 4);
    expect(map.size).toBe(3);
    expect(map.has("a")).toBe(false);
    expect(Array.from(map.keys())).toEqual(["b", "c", "d"]);
  });

  it("CappedArray evicts oldest elements when exceeding max size", () => {
    const arr = new CappedArray<number>(3);
    arr.push(1);
    arr.push(2);
    arr.push(3);
    expect(arr.length).toBe(3);
    expect(arr.raw).toEqual([1, 2, 3]);

    // Push 4th element -> oldest (1) is evicted
    arr.push(4);
    expect(arr.length).toBe(3);
    expect(arr.raw).toEqual([2, 3, 4]);

    // Push batch
    arr.push(5, 6);
    expect(arr.length).toBe(3);
    expect(arr.raw).toEqual([4, 5, 6]);
  });

  it("calculateStrategyScore returns deterministic scores between 0 and 100", () => {
    const score1 = calculateStrategyScore({
      concept: "Fractions",
      strategy: "visual",
      correct: 1,
      timeTaken: 10,
      attemptCount: 1,
      hintUsed: 0,
    });
    expect(score1).toBe(100);

    const score2 = calculateStrategyScore({
      concept: "Fractions",
      strategy: "visual",
      correct: 0,
      timeTaken: 40,
      attemptCount: 3,
      hintUsed: 1,
    });
    expect(score2).toBe(5);
  });

  it("recordStrategyObservationsBatch calculates score once and writes to fallback store when DB is absent", async () => {
    resetInMemoryStoresForTesting();

    const observations = [
      { concept: "c1", strategy: "visual", correct: 1, timeTaken: 10, attemptCount: 1, hintUsed: 0 },
      { concept: "c2", strategy: "step-by-step", correct: 1, timeTaken: 20, attemptCount: 1, hintUsed: 0 },
    ];

    const result = await recordStrategyObservationsBatch(5001, "Fractions", observations);
    expect(result.preferredStrategy).toBeDefined();
    expect(result.strategyScores).toBeDefined();

    // Since in unit tests no live MySQL DB exists, it successfully stores in inMemory stores
    const counts = getInMemoryRecordCounts();
    expect(counts.strategyObservations).toBe(2);
    expect(counts.strategyProfiles).toBe(1);
  });
});
