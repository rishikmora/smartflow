/**
 * Server-side proxy to the five SmartFlow services.
 *
 * The browser never talks to a service directly and never sees a bearer token.
 * It calls this route, which mints a development token (or forwards an Auth0
 * one), calls the service over the internal network, and returns the JSON.
 *
 * That keeps the token server-side and means the five services need no CORS
 * configuration between them.
 */

import { NextRequest, NextResponse } from "next/server";
import { createHmac } from "crypto";

const SERVICES: Record<string, string | undefined> = {
  sim: process.env.SIM_SERVICE_URL,
  rl: process.env.RL_SERVICE_URL,
  vision: process.env.VISION_SERVICE_URL,
  graph: process.env.GRAPH_SERVICE_URL,
  llm: process.env.LLM_SERVICE_URL,
};

const DEV_SECRET = process.env.SMARTFLOW_DEV_JWT_SECRET ?? "smartflow-dev-secret";

function base64url(input: Buffer | string): string {
  return Buffer.from(input)
    .toString("base64")
    .replace(/=/g, "")
    .replace(/\+/g, "-")
    .replace(/\//g, "_");
}

/**
 * Mint the same HS256 development token the Python services accept.
 *
 * Deliberately hand-rolled rather than pulling a JWT library in: it is a fixed,
 * known claim set, and the services' own auth module is the authority on the
 * format. A real deployment sets AUTH0_* and forwards the user's token instead.
 */
function devToken(): string {
  const now = Math.floor(Date.now() / 1000);
  const header = base64url(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = base64url(
    JSON.stringify({
      sub: "dashboard",
      aud: "smartflow-dev",
      iat: now,
      exp: now + 3600,
      scope: "read:metrics read:graph read:vision",
    })
  );
  const signature = base64url(
    createHmac("sha256", DEV_SECRET).update(`${header}.${payload}`).digest()
  );
  return `${header}.${payload}.${signature}`;
}

export async function GET(request: NextRequest) {
  const service = request.nextUrl.searchParams.get("service");
  const path = request.nextUrl.searchParams.get("path") ?? "/health";

  if (!service || !(service in SERVICES)) {
    return NextResponse.json(
      { error: `Unknown service. Known: ${Object.keys(SERVICES).join(", ")}` },
      { status: 400 }
    );
  }
  const base = SERVICES[service];
  if (!base) {
    return NextResponse.json(
      { error: `${service} has no configured URL` },
      { status: 503 }
    );
  }
  // Only allow paths the dashboard actually uses, so this proxy cannot be
  // turned into an open relay to anything on the internal network.
  if (!path.startsWith("/")) {
    return NextResponse.json({ error: "Path must start with /" }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${base}${path}`, {
      headers: { Authorization: `Bearer ${devToken()}` },
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    });
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `${service} unreachable: ${String(error)}` },
      { status: 502 }
    );
  }
}

export async function POST(request: NextRequest) {
  const service = request.nextUrl.searchParams.get("service");
  const path = request.nextUrl.searchParams.get("path");

  if (service !== "llm" || path !== "/query") {
    return NextResponse.json(
      { error: "Only the analytics query endpoint accepts POST." },
      { status: 400 }
    );
  }
  const base = SERVICES.llm;
  if (!base) {
    return NextResponse.json({ error: "llm service has no URL" }, { status: 503 });
  }

  try {
    const payload = await request.json();
    const upstream = await fetch(`${base}/query`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${devToken()}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `llm unreachable: ${String(error)}` },
      { status: 502 }
    );
  }
}
