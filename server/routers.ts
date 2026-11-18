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
    // What strategy should this topic's *current* lesson use? Priority:
    // 1. The ML loop's last decision for this topic (lessonPerformanceEvents),
    //    if the student has already had at least one real lesson here.
    // 2. This topic's onboarding-derived cold-start profile.
    // 3. The onboarding baseline quiz only calibrates one topic per subject,
    //    so most topics never get their own cold-start profile -- fall back
    //    to the student's most recently derived profile from another topic.
    // 4. null (caller uses its own hardcoded default -- we never fabricate
    //    a profile here).
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
    // The normal adaptive loop: record how the just-finished lesson went,
    // ask the ML strategy service what to teach next, and persist both. If
    // the ML service is unreachable, falls back to the strategy that was
    // just used (per the brief: never crash the lesson, never guess
    // randomly) rather than surfacing an error to the student.
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

        const mlResult = await callMLPredictStrategy({
          previousAccuracy: input.accuracy,
          responseTime: input.responseTime,
          attempts: input.attempts,
          hintsRequested: input.hintsRequested,
          topicMastery: input.topicMastery,
          previousStrategy: input.previousStrategy,
        });

        const nextStrategy = mlResult?.nextStrategy ?? input.previousStrategy;
        const source = mlResult ? ("ml" as const) : ("fallback" as const);

        await recordLessonPerformance(userId, input.topic, {
          strategyUsed: input.previousStrategy,
          accuracy: input.accuracy,
          responseTime: input.responseTime,
          attempts: input.attempts,
          hintsRequested: input.hintsRequested,
          topicMastery: input.topicMastery,
          predictedNextStrategy: mlResult?.nextStrategy ?? null,
          predictedConfidence: mlResult?.confidence ?? null,
        });

        return { nextStrategy, confidence: mlResult?.confidence ?? null, source };
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
          console.error("[AI Tutor] Falling back to structured educational tutor response:", error?.message);
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
          targetStrategy = profile?.preferredStrategy || "step-by-step";
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
            adaptedExplanation: adaptedExplanation || `Let's break this down differently: ${input.currentExplanation}`,
            strategyUsed: targetStrategy,
          };
        } catch (err: any) {
          console.warn("[AI Adapt] LLM call failed, using fallback:", err?.message);
          return {
            adaptedExplanation: `Here's a simpler way to think about ${input.lessonTitle}:\n\n1. Take a breath — let's look at the core idea.\n2. ${input.currentExplanation.slice(0, 150)}...\n3. Try breaking it into small pieces step by step!`,
            strategyUsed: targetStrategy,
          };
        }
      }),
  }),
});

export type AppRouter = typeof appRouter;

