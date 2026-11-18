import { COOKIE_NAME } from "@shared/const";
import { z } from "zod";
import { getSessionCookieOptions } from "./_core/cookies";
import { ENV } from "./_core/env";
import { invokeLLM, type Message } from "./_core/llm";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { TRPCError } from "@trpc/server";
import {
  getDashboardData,
  getLatestLessonPerformance,
  getMostRecentStrategyProfile,
  getStudentStrategyProfile,
  recordLearningMode,
  recordLessonPerformance,
  recordStrategyObservationsBatch,
  recordStudySession,
  resetDashboardData,
  setTaskStatus,
} from "./db";
import { buildTutorSystemPrompt, generateFallbackTutorReply } from "./aiPrompts";
import { predictNextStrategy as callMLPredictStrategy } from "./_core/mlClient";

const STRATEGY_VALUES = ["visual", "textual", "step-by-step", "example-based", "socratic"] as const;

function effectiveUserId(user: { id: number } | null | undefined) {
  // Guests get an isolated empty workspace (id 0), never another user's seeded data.
  return user?.id ?? 0;
}

export const MIN_CONFIDENCE_THRESHOLD = 0.5;

export const predictionMetrics = {
  totalRequests: 0,
  realMlSuccess: 0,
  fallbackCount: 0,
  lowConfidenceCount: 0,
  byReason: {} as Record<string, number>,
};

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  dashboard: router({
    overview: publicProcedure.query(async ({ ctx }) => getDashboardData(effectiveUserId(ctx.user))),
    resetProgress: publicProcedure.mutation(async ({ ctx }) => {
      await resetDashboardData(effectiveUserId(ctx.user));
      return { success: true } as const;
    }),
    completeTask: publicProcedure
      .input(z.object({ taskId: z.number().int().positive(), completed: z.boolean() }))
      .mutation(async ({ ctx, input }) => {
        await setTaskStatus(effectiveUserId(ctx.user), input.taskId, input.completed);
        return { success: true } as const;
      }),
    logStudy: publicProcedure
      .input(z.object({ courseId: z.number().int().positive(), minutes: z.number().int().min(5).max(240) }))
      .mutation(async ({ ctx, input }) => {
        await recordStudySession(effectiveUserId(ctx.user), input.courseId, input.minutes);
        return { success: true } as const;
      }),
    recordLearningMode: publicProcedure
      .input(z.object({ courseId: z.number().int().positive(), topic: z.string().min(1), mode: z.string().min(1).max(40) }))
      .mutation(async ({ ctx, input }) => {
        await recordLearningMode(effectiveUserId(ctx.user), input.courseId, input.topic, input.mode);
        return { success: true } as const;
      }),
  }),
  adaptive: router({
    submitBaselineAssessment: publicProcedure
      .input(
        z.object({
          topic: z.string().min(1),
          assessmentId: z.string().optional(),
          observations: z.array(
            z.object({
              lessonId: z.number().optional(),
              concept: z.string().min(1),
              strategy: z.string().min(1),
              questionDifficulty: z.string().optional(),
              correct: z.number().min(0).max(1),
              score: z.number().optional(),
              timeTaken: z.number().min(0),
              attemptCount: z.number().min(1).optional(),
              hintUsed: z.number().min(0).max(1).optional(),
            })
          ),
        })
      )
      .mutation(async ({ ctx, input }) => {
        const userId = effectiveUserId(ctx.user);
        const result = await recordStrategyObservationsBatch(
          userId,
          input.topic,
          input.observations.map((obs) => ({
            ...obs,
            assessmentId: input.assessmentId,
          }))
        );
        return {
          success: true,
          ...result,
        };
      }),
    getStrategyProfile: publicProcedure
      .input(z.object({ topic: z.string().min(1) }))
      .query(async ({ ctx, input }) => {
        const userId = effectiveUserId(ctx.user);
        const profile = await getStudentStrategyProfile(userId, input.topic);
        return { profile };
      }),
    getInitialStrategy: publicProcedure
      .input(z.object({ topic: z.string().min(1) }))
      .query(async ({ ctx, input }) => {
        const userId = effectiveUserId(ctx.user);
        const averageScore = (scores: Record<string, number>): number | null => {
          const values = Object.values(scores);
          if (values.length === 0) return null;
          return values.reduce((sum, v) => sum + v, 0) / values.length / 100;
        };

        const lastEvent = await getLatestLessonPerformance(userId, input.topic);
        if (lastEvent) {
          const strategy = lastEvent.predictedNextStrategy ?? lastEvent.strategyUsed;
          return { strategy, topicMastery: lastEvent.topicMastery, source: "ml-loop" as const };
        }

        const exact = await getStudentStrategyProfile(userId, input.topic);
        if (exact) {
          return {
            strategy: exact.preferredStrategy,
            topicMastery: averageScore(exact.strategyScores),
            source: "onboarding" as const,
          };
        }

        const fallback = await getMostRecentStrategyProfile(userId);
        if (fallback) {
          return {
            strategy: fallback.preferredStrategy,
            topicMastery: averageScore(fallback.strategyScores),
            source: "cross-topic" as const,
          };
        }

        return { strategy: null, topicMastery: null, source: "none" as const };
      }),
    predictNextStrategy: publicProcedure
      .input(
        z.object({
          topic: z.string().min(1),
          previousStrategy: z.enum(STRATEGY_VALUES),
          accuracy: z.number().min(0).max(1),
          responseTime: z.number().gt(0),
          attempts: z.number().int().min(1),
          hintsRequested: z.number().int().min(0),
          topicMastery: z.number().min(0).max(1),
        })
      )
      .mutation(async ({ ctx, input }) => {
        const userId = effectiveUserId(ctx.user);
        predictionMetrics.totalRequests++;

        const mlResult = await callMLPredictStrategy({
          previousAccuracy: input.accuracy,
          responseTime: input.responseTime,
          attempts: input.attempts,
          hintsRequested: input.hintsRequested,
          topicMastery: input.topicMastery,
          previousStrategy: input.previousStrategy,
        });

        let nextStrategy: typeof STRATEGY_VALUES[number] = input.previousStrategy;
        let confidence: number | null = null;
        let source: "ml" | "fallback" | "low-confidence" = "fallback";

        if (!mlResult.ok) {
          predictionMetrics.fallbackCount++;
          predictionMetrics.byReason[mlResult.reason] = (predictionMetrics.byReason[mlResult.reason] || 0) + 1;
          console.warn("[ML Fallback]", {
            trigger: "predictNextStrategy",
            fallbackReason: mlResult.reason,
            status: mlResult.status,
            error: mlResult.error,
            previousStrategy: input.previousStrategy,
            topic: input.topic,
            metrics: { ...predictionMetrics },
          });
          nextStrategy = input.previousStrategy;
          confidence = null;
          source = "fallback";
        } else if (mlResult.data.confidence < MIN_CONFIDENCE_THRESHOLD) {
          predictionMetrics.lowConfidenceCount++;
          console.warn("[ML Fallback: Low Confidence]", {
            trigger: "predictNextStrategy",
            confidence: mlResult.data.confidence,
            threshold: MIN_CONFIDENCE_THRESHOLD,
            predictedStrategy: mlResult.data.nextStrategy,
            fallbackStrategy: input.previousStrategy,
            topic: input.topic,
            metrics: { ...predictionMetrics },
          });
          nextStrategy = input.previousStrategy;
          confidence = mlResult.data.confidence;
          source = "low-confidence";
        } else {
          predictionMetrics.realMlSuccess++;
          nextStrategy = mlResult.data.nextStrategy;
          confidence = mlResult.data.confidence;
          source = "ml";
        }

        console.info("[ML Live Outcome]", {
          userId,
          topic: input.topic,
          strategyUsed: input.previousStrategy,
          actualOutcome: {
            accuracy: input.accuracy,
            responseTime: input.responseTime,
            attempts: input.attempts,
            hintsRequested: input.hintsRequested,
            topicMastery: input.topicMastery,
          },
          predictedNextStrategy: nextStrategy,
          predictedConfidence: confidence,
          source,
          timestamp: new Date().toISOString(),
        });

        await recordLessonPerformance(userId, input.topic, {
          strategyUsed: input.previousStrategy,
          accuracy: input.accuracy,
          responseTime: input.responseTime,
          attempts: input.attempts,
          hintsRequested: input.hintsRequested,
          topicMastery: input.topicMastery,
          predictedNextStrategy: nextStrategy,
          predictedConfidence: confidence,
        });

        return { nextStrategy, confidence, source };
      }),
  }),
  ai: router({
    tutorChat: publicProcedure
      .input(
        z.object({
          topic: z.string(),
          lessonTitle: z.string(),
          lessonContent: z.string(),
          currentMode: z.string().optional(),
          currentQuestion: z
            .object({
              question: z.string(),
              options: z.array(z.string()).optional(),
            })
            .optional(),
          conversation: z.array(
            z.object({
              role: z.enum(["user", "assistant"]),
              content: z.string(),
            })
          ),
          message: z.string().min(1),
        })
      )
      .mutation(async ({ input }) => {
        if (!ENV.forgeApiKey) {
          return { reply: generateFallbackTutorReply(input) };
        }

        try {
          const systemPrompt = buildTutorSystemPrompt(input);

          const messages: Message[] = [
            { role: "system", content: systemPrompt },
            ...input.conversation.map((msg) => ({
              role: msg.role as "user" | "assistant",
              content: msg.content,
            })),
            { role: "user", content: input.message },
          ];

          const response = await invokeLLM({ messages });
          const firstChoice = response.choices?.[0]?.message?.content;
          let reply = "";
          if (typeof firstChoice === "string") {
            reply = firstChoice;
          } else if (Array.isArray(firstChoice)) {
            reply = firstChoice
              .map((part) => (part.type === "text" ? part.text : ""))
              .join("");
          }

          if (!reply.trim()) {
            reply =
              "I'm here to help! Could you please ask that in another way or specify which part of the lesson you'd like to explore?";
          }

          return { reply };
        } catch (error: any) {
          console.warn("[AI Tutor] LLM request failed, using structured educational tutor fallback:", error?.message);
          return { reply: generateFallbackTutorReply(input) };
        }
      }),
    adaptTeachingTechnique: publicProcedure
      .input(
        z.object({
          topic: z.string(),
          lessonTitle: z.string(),
          currentExplanation: z.string(),
          currentStrategy: z.string().optional(),
          preferredStrategy: z.string().optional(),
        })
      )
      .mutation(async ({ ctx, input }) => {
        const userId = effectiveUserId(ctx.user);
        let targetStrategy = input.preferredStrategy;
        if (!targetStrategy) {
          const profile = await getStudentStrategyProfile(userId, input.topic);
          targetStrategy = profile?.preferredStrategy || "visual";
        }

        const buildFallbackExplanation = (strategy: string) => {
          const snippet = input.currentExplanation.length > 20
            ? input.currentExplanation.slice(0, 180).trim()
            : `Understanding how ${input.lessonTitle} works in ${input.topic}`;

          switch (strategy) {
            case "step-by-step":
              return `### Step-by-Step Breakdown: ${input.lessonTitle}\n\nLet's break this down into 3 clear consecutive micro-steps:\n\n1. **Step 1: Notice the core components** — Focus on the initial conditions given in the problem.\n2. **Step 2: Apply the key rule** — ${snippet}.\n3. **Step 3: Connect to the result** — Put the pieces together to find your answer!\n\nTake it one step at a time!`;
            case "example-based":
              return `### Real-World Analogy: ${input.lessonTitle}\n\nThink about **${input.lessonTitle}** through an everyday example:\n\n* **Everyday Analogy:** Imagine sharing slices of pizza equally with friends or pouring water into measuring cups. The exact same balance applies here.\n* **Applying the concept:** ${snippet}.\n* **Takeaway:** Whenever you feel stuck, picture this physical model in your mind!`;
            case "socratic":
              return `### Guided Discovery: ${input.lessonTitle}\n\nLet's think through this together with two gentle questions:\n\n1. **Question 1:** What clue is given first, and which part is asking for your answer?\n2. **Question 2:** If we apply the rule (${snippet}), what should happen next?\n\nTake a moment to reflect on those two clues — you've got this!`;
            case "textual":
              return `### Clear Structured Summary: ${input.lessonTitle}\n\n* **Core Definition:** ${snippet}.\n* **Why It Works:** Changing one variable creates a direct, predictable change in the outcome.\n* **Key Takeaway:** Focus on the fundamental definition before calculating.`;
            case "visual":
            default:
              return `### Visual Mental Model: ${input.lessonTitle}\n\n\`\`\`\n[ Starting State ]  ===>  [ Core Relationship ]  ===>  [ Solution ]\n\`\`\`\n\n* **Key Relationship:** ${snippet}.\n* **Mental Model:** Picture the elements flowing from left to right in sequence.`;
          }
        };

        if (!ENV.forgeApiKey) {
          return {
            adaptedExplanation: buildFallbackExplanation(targetStrategy),
            strategyUsed: targetStrategy,
          };
        }

        const prompt = `You are an adaptive AI learning tutor for Neura. The student is currently stuck on the concept "${input.lessonTitle}" in "${input.topic}".
Current explanation: "${input.currentExplanation}".
Your job is to re-explain this exact concept using the "${targetStrategy}" pedagogical strategy:
- visual: Use clear spatial cues, ASCII art diagrams, structured boxes, or shape-based mental models.
- step-by-step: Break down the logic into numbered, bite-sized consecutive micro-steps (Step 1, Step 2, Step 3).
- example-based: Use relatable real-world analogies (e.g. sharing pizza slices, water pipes, sports, recipes).
- textual: Provide clean, dyslexia-friendly structured bullet points with high semantic clarity.
- socratic: Ask 2 guiding question prompts with gentle hints that help them discover the answer themselves.

Provide a warm, reassuring, concise re-explanation. Keep under 150 words.`;

        try {
          const response = await invokeLLM({
            messages: [
              { role: "system", content: "You are an empathetic, encouraging adaptive learning tutor." },
              { role: "user", content: prompt },
            ],
          });
          const content = response.choices?.[0]?.message?.content;
          const adaptedExplanation = typeof content === "string" ? content : (Array.isArray(content) ? content.map(p => (p as any).text || "").join("") : "");
          return {
            adaptedExplanation: adaptedExplanation || buildFallbackExplanation(targetStrategy),
            strategyUsed: targetStrategy,
          };
        } catch (err: any) {
          console.warn("[AI Adapt] LLM call failed, using strategy-specific fallback:", err?.message);
          return {
            adaptedExplanation: buildFallbackExplanation(targetStrategy),
            strategyUsed: targetStrategy,
          };
        }
      }),
  }),
});

export type AppRouter = typeof appRouter;

