import "dotenv/config";

const rawApiKey =
  process.env.BUILT_IN_FORGE_API_KEY ||
  process.env.OPENAI_API_KEY ||
  "";

const isGroq = rawApiKey.startsWith("gsk_");

export const ENV = {
  appId: process.env.VITE_APP_ID ?? "neura",
  cookieSecret: process.env.JWT_SECRET || "development-jwt-secret-key-must-be-at-least-32-chars",
  databaseUrl: process.env.DATABASE_URL ?? "",
  ownerOpenId: process.env.OWNER_OPEN_ID ?? "",
  isProduction: process.env.NODE_ENV === "production",
  forgeApiUrl:
    process.env.BUILT_IN_FORGE_API_URL ||
    process.env.OPENAI_BASE_URL ||
    (isGroq ? "https://api.groq.com/openai/v1" : rawApiKey ? "https://api.openai.com" : ""),
  forgeApiKey: rawApiKey,
  defaultModel:
    process.env.OPENAI_MODEL ||
    (isGroq ? "openai/gpt-oss-120b" : "gpt-4o-mini"),
  mlServiceUrl: process.env.ML_SERVICE_URL || "http://localhost:8001",
};
