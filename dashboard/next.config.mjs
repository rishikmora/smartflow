/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The dashboard talks to the five services through its own /api routes rather
  // than from the browser directly. That keeps the bearer token server-side and
  // avoids CORS configuration on five separate services.
  env: {
    SIM_SERVICE_URL: process.env.SIM_SERVICE_URL ?? "http://localhost:8001",
    RL_SERVICE_URL: process.env.RL_SERVICE_URL ?? "http://localhost:8002",
    VISION_SERVICE_URL: process.env.VISION_SERVICE_URL ?? "http://localhost:8003",
    GRAPH_SERVICE_URL: process.env.GRAPH_SERVICE_URL ?? "http://localhost:8004",
    LLM_SERVICE_URL: process.env.LLM_SERVICE_URL ?? "http://localhost:8005",
  },
};
export default nextConfig;
