export interface TutorSystemPromptInput {
  topic: string;
  lessonTitle: string;
  lessonContent: string;
  currentMode?: string;
  currentQuestion?: {
    question: string;
    options?: string[];
  };
}

export const DEFAULT_TUTOR_SYSTEM_PROMPT = `You are the Neura AI Tutor, an empathetic, engaging, and highly pedagogical adaptive learning assistant.
Your goal is to help the student deeply understand the lesson through interactive dialogue, tailored explanations, and guided discovery.

PEDAGOGICAL GUIDELINES:
1. Teach conversationally rather than providing rigid or repetitive lectures.
2. Adapt to the student's questions and emotional tone (e.g., if they say "I don't understand", meet them with reassurance, simplify the concept, and provide a fresh perspective or analogy).
3. Break down complex ideas into bite-sized, digestible intuitive steps.
4. Use rich real-world analogies, mental models, and concrete examples.
5. If the student is stuck on a quiz question, offer guiding clues and intuitive hints—NEVER give the letter or raw answer directly.
6. Keep responses friendly, supportive, well-formatted, and concise (avoid overwhelming walls of text).`;

/**
 * Builds a dynamic, contextual system prompt for the Neura AI Tutor
 * tailored to the current topic, lesson, learning mode, and active quiz question.
 */
export function buildTutorSystemPrompt(input: TutorSystemPromptInput): string {
  const modeInstructions: Record<string, string> = {
    visual: "VISUAL / MENTAL MODEL STYLE: Use spatial cues, ASCII art diagrams, structured boxes, or physical mental models to make abstract concepts visual.",
    textual: "TEXTUAL / CONCEPTUAL STYLE: Provide clear, structured, rich conceptual explanations with precise definitions and logical flow.",
    "step-by-step": "STEP-BY-STEP STYLE: Break down explanations into numbered, clear consecutive micro-steps (Step 1, Step 2, Step 3).",
    steps: "STEP-BY-STEP STYLE: Break down explanations into numbered, clear consecutive micro-steps (Step 1, Step 2, Step 3).",
    "example-based": "EXAMPLE-BASED / ANALOGY STYLE: Anchor your explanations in intuitive, relatable daily life scenarios (e.g. food sharing, sports, games, nature, technology) and concrete worked examples.",
    example: "EXAMPLE-BASED / ANALOGY STYLE: Anchor your explanations in intuitive, relatable daily life scenarios (e.g. food sharing, sports, games, nature, technology) and concrete worked examples.",
    simple: "BEGINNER / SIMPLE STYLE: Use plain, non-intimidating language, friendly tone, and zero unnecessary jargon.",
    socratic: "SOCRATIC / GUIDED STYLE: Lead the student with 1-2 thoughtful guiding questions and gentle clues so they reach the insight themselves.",
    stuck: "EMPATHETIC INTERVENTION STYLE: Reassure the student, strip away cognitive overload, and explain the core principle in the simplest, most intuitive way.",
  };

  const currentModeInstruction = input.currentMode && modeInstructions[input.currentMode]
    ? modeInstructions[input.currentMode]
    : "ADAPTIVE TUTOR STYLE: Blend clear step-by-step reasoning with intuitive examples and supportive check-ins.";

  const parts: string[] = [
    `You are the Neura AI Tutor, an empathetic, expert adaptive learning assistant.`,
    `Your mission is to help the student master the lesson in an engaging, interactive, and encouraging way.`,
    `CURRENT LESSON CONTEXT:`,
    `- Topic: ${input.topic}`,
    `- Lesson Title: ${input.lessonTitle}`,
    `- Lesson Content:\n${input.lessonContent}`,
    input.currentMode ? `- Current Learning Mode: ${input.currentMode}\n- Mode Guideline: ${currentModeInstruction}` : "",
    input.currentQuestion
      ? `- Active Quiz Question: "${input.currentQuestion.question}"${
          input.currentQuestion.options && input.currentQuestion.options.length > 0
            ? ` (Options: ${input.currentQuestion.options.join(", ")})`
            : ""
        }`
      : "",
    `PEDAGOGICAL GUIDELINES:`,
    `1. EMOTIONAL & COGNITIVE RESPONSIVENESS: If the student says "I don't understand", "what does this mean?", "I'm lost", or expresses confusion, NEVER repeat the same lesson text verbatim. Acknowledge their struggle with warmth, simplify the concept to its bare essence, and use a fresh analogy or visual mental model.`,
    `2. ACTIVE LEARNING: Don't just lecture. Invite the student to try the next step, ask what part makes sense so far, or give a friendly check-in question.`,
    `3. FORMATTING: Use clean markdown with bold highlights, short paragraphs, and bullet points for readability.`,
    `4. QUIZ HANDLING: If the student asks for help with a quiz question or seems stuck: NEVER give away the correct answer or choice directly. Identify the core concept being tested, give a gentle hint, and encourage them to reason through the options.`,
  ];

  return parts.filter(Boolean).join("\n\n");
}

export interface FallbackTutorInput {
  topic: string;
  lessonTitle?: string;
  lessonContent?: string;
  currentMode?: string;
  currentQuestion?: {
    question: string;
    options?: string[];
  };
  message: string;
}

/**
 * Generates a helpful, structured educational reply when the live LLM is offline or no API key is provided.
 * Ensures the student can always communicate and learn without seeing a raw server error.
 */
export function generateFallbackTutorReply(input: FallbackTutorInput): string {
  const query = input.message.toLowerCase().trim();
  const topic = input.topic || "this topic";
  const lessonTitle = input.lessonTitle || topic;
  const content = input.lessonContent || "";

  // 1. Quiz / Question hint handling
  if (
    input.currentQuestion &&
    (query.includes("answer") ||
      query.includes("hint") ||
      query.includes("stuck") ||
      query.includes("help with question") ||
      query.includes("quiz"))
  ) {
    const qText = input.currentQuestion.question;
    return `### 💡 AI Tutor Hint: ${lessonTitle}\n\nLet's break down this question together:\n\n> *"${qText}"*\n\n**Key Clue to Guide You:**\n* Focus on the core rule we just covered in **${lessonTitle}**.\n* What happens when the main condition changes? Which part directly controls the outcome?\n* Rule out choices that contradict the fundamental concept.\n\nTake another look at the choices — which one feels like it matches this clue best?`;
  }

  // 2. "Don't understand" / Confused / Stuck handling
  if (
    query.includes("don't understand") ||
    query.includes("dont understand") ||
    query.includes("not understand") ||
    query.includes("confused") ||
    query.includes("lost") ||
    query.includes("hard") ||
    query.includes("difficult")
  ) {
    return `### 🌟 Let's Make ${topic} Crystal Clear!\n\nNo worries at all! It's completely normal to take a moment to click with **${lessonTitle}**.\n\nLet's step back and look at it from a fresh angle:\n\n1. **The Big Picture:** At its heart, **${lessonTitle}** is just about comparing or connecting two key pieces.\n2. **The Golden Rule:** When you look at the parts, ask yourself: *which piece is making things bigger or smaller, and which stays the same?*\n3. **A Simple Picture:** Imagine holding two measuring cups or dividing a pizza — having bigger pieces means fewer slices, while having more slices means each piece is smaller!\n\nDoes this picture make more sense, or would you like to see a quick step-by-step example?`;
  }

  // 3. Simple explanation request
  if (query.includes("simply") || query.includes("beginner") || query.includes("simple") || query.includes("easy")) {
    let contextSnippet = "";
    if (content && content.length > 30) {
      const firstSentence = content.split(/[.\n]/).filter(s => s.trim().length > 15)[0] || "";
      if (firstSentence) {
        contextSnippet = `\n\n* **Core Idea from the Lesson:** ${firstSentence.trim()}.`;
      }
    }

    return `### 🧩 Understanding ${topic} Simply\n\nAt its core, **${topic}** is all about understanding how key ideas connect and work together.${contextSnippet}\n\nHere is a gentle way to think about it:\n1. **The Core Concept**: Think of **${topic}** as a system where every part has a clear, specific role.\n2. **Why It Matters**: It helps us understand, predict, and solve real-world questions that would otherwise feel complex.\n3. **Quick Analogy**: Imagine building with interlocking blocks — once you see how the foundational pieces fit together, the entire structure makes sense.\n\nWhat specific part of **${topic}** would you like us to look into next?`;
  }

  // 4. 3 Core Principles request
  if (query.includes("principle") || query.includes("rule") || query.includes("core")) {
    return `### 📐 3 Core Principles of ${topic}\n\n1. **Foundations First**: In **${topic}**, every advanced concept builds directly upon simple, foundational definitions.\n2. **Cause & Effect**: Changing an input or condition leads to predictable, observable changes across the system.\n3. **Practical Application**: Concepts in **${topic}** aren't just abstract theory — they provide frameworks used to solve real-world challenges.\n\nWhich of these three principles would you like to explore deeper?`;
  }

  // 5. Real-world example request
  if (query.includes("example") || query.includes("real-world") || query.includes("real world") || query.includes("analogy")) {
    return `### 🌍 Real-World Example of ${topic}\n\nA great way to picture **${topic}** in everyday life is through real-world systems:\n\n* **Everyday Observation**: Notice how patterns, balances, and systems around us rely on the exact principles of **${topic}**.\n* **Practical Impact**: Engineers, scientists, and researchers apply **${topic}** daily to build better tools, understand environments, and make informed choices.\n\nWould you like an example from nature, modern technology, or everyday human activities?`;
  }

  // 6. Step-by-step breakdown request
  if (query.includes("step") || query.includes("breakdown") || query.includes("break down") || query.includes("how does it work")) {
    return `### 📝 Step-by-Step Breakdown of ${topic}\n\nHere is how we deconstruct **${topic}** into bite-sized steps:\n\n* **Step 1 — Understand the Goal**: Identify what question or problem **${topic}** is addressing.\n* **Step 2 — Identify the Components**: Break the topic into its individual elements or variables.\n* **Step 3 — Observe the Interactions**: See how the pieces communicate, calculate, or react with one another.\n* **Step 4 — Verify & Reflect**: Connect what you observed back to the big picture.\n\nWhich step shall we dive into together?`;
  }

  // 7. Understanding check / quiz request
  if (query.includes("check") || query.includes("quiz") || query.includes("test")) {
    return `### 🎯 Quick Understanding Check on ${topic}\n\nHere is a gentle question to check your thoughts on **${topic}**:\n\n*If you had to explain the main purpose of **${topic}** to a friend in one single sentence, what would you say is the most important thing it helps us do?*\n\nTake your time — there's no rush!`;
  }

  // 8. General friendly pedagogical tutor response
  return `### 💬 Exploring ${topic} with your AI Tutor\n\nThat's a thoughtful question about **${topic}**!\n\nWhen exploring "${input.message}", the key thing to keep in mind is how **${topic}** connects basic observations to structured understanding.\n\n* **Key Takeaway**: Breaking down complex ideas into manageable observations always makes **${topic}** easier to grasp.\n* **Next Step**: We can explore a real-world example, look at step-by-step principles, or answer any specific questions you have.\n\nWhat would feel most helpful to focus on right now?`;
}
