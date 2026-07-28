/**
 * Hermes Bridge Registry — Cloudflare Worker
 * 
 * Maps user emails to their current bridge tunnel URL.
 * Uses Workers KV for persistence (free tier).
 *
 * KV Namespace binding: HERMES_REGISTRY
 *
 * Endpoints:
 *   POST /api/v1/register       — one-time bridge registration
 *   POST /api/v1/heartbeat      — periodic URL update (every 3 min)
 *   GET  /api/v1/discover/:email — app discovers bridge URL
 */

// ─── SHA-256 helper (Web Crypto API) ──────────────────────────────

async function sha256(message) {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ─── CORS headers ──────────────────────────────────────────────────

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, 'Content-Type': 'application/json' },
  });
}

// ─── Handlers ──────────────────────────────────────────────────────

/**
 * POST /api/v1/register
 * One-time registration of a bridge device.
 * Creates a KV entry keyed by SHA-256(email).
 */
async function handleRegister(request, env) {
  const body = await request.json();
  const { email, device_id, device_secret, device_name } = body;

  if (!email || !device_id || !device_secret) {
    return json({ error: 'email, device_id, device_secret are required' }, 400);
  }

  const emailHash = await sha256(email.toLowerCase().trim());
  const secretHash = await sha256(device_secret);
  const key = `email:${emailHash}`;

  // Check if already registered
  const existing = await env.HERMES_REGISTRY.get(key, 'json');
  if (existing) {
    return json({ error: 'Email already registered', device_id: existing.device_id }, 409);
  }

  const record = {
    device_id,
    device_secret_hash: secretHash,
    device_name: device_name || 'Unknown Device',
    tunnel_url: null,
    platform: null,
    version: null,
    last_seen: Date.now(),
    created_at: Date.now(),
  };

  await env.HERMES_REGISTRY.put(key, JSON.stringify(record));

  // Also store reverse lookup: device_id → emailHash
  await env.HERMES_REGISTRY.put(`device:${device_id}`, emailHash);

  return json({ status: 'ok', device_id });
}

/**
 * POST /api/v1/heartbeat
 * Called every ~3 minutes by the bridge server.
 * Updates tunnel URL and last_seen timestamp.
 * Authenticated via device_secret (SHA-256 compared against stored hash).
 */
async function handleHeartbeat(request, env) {
  const body = await request.json();
  const { device_id, device_secret, tunnel_url, platform, version, device_name } = body;

  if (!device_id || !device_secret) {
    return json({ error: 'device_id and device_secret are required' }, 400);
  }

  // Find the email hash via reverse lookup
  const emailHash = await env.HERMES_REGISTRY.get(`device:${device_id}`);
  if (!emailHash) {
    return json({ error: 'Device not registered' }, 404);
  }

  const key = `email:${emailHash}`;
  const record = await env.HERMES_REGISTRY.get(key, 'json');
  if (!record) {
    return json({ error: 'Record not found (inconsistent state)' }, 500);
  }

  // Verify device_secret
  const secretHash = await sha256(device_secret);
  if (secretHash !== record.device_secret_hash) {
    return json({ error: 'Invalid device secret' }, 401);
  }

  // Update record
  record.tunnel_url = tunnel_url ?? record.tunnel_url;
  record.platform = platform ?? record.platform;
  record.version = version ?? record.version;
  record.device_name = device_name ?? record.device_name;
  record.last_seen = Date.now();

  await env.HERMES_REGISTRY.put(key, JSON.stringify(record));

  return json({ status: 'ok', last_seen: record.last_seen });
}

/**
 * GET /api/v1/discover/:email
 * Public endpoint — mobile app discovers its bridge URL.
 * No auth required since the URL is already public (Cloudflare tunnel).
 */
async function handleDiscover(email, env) {
  if (!email) {
    return json({ error: 'Email is required' }, 400);
  }

  const emailHash = await sha256(decodeURIComponent(email).toLowerCase().trim());
  const key = `email:${emailHash}`;
  const record = await env.HERMES_REGISTRY.get(key, 'json');

  if (!record) {
    return json({ error: 'No bridge found for this email' }, 404);
  }

  // Consider offline if last heartbeat was >10 minutes ago
  const online = (Date.now() - record.last_seen) < 10 * 60 * 1000;

  return json({
    url: record.tunnel_url,
    device_name: record.device_name,
    device_id: record.device_id,
    online,
    last_seen: record.last_seen,
    platform: record.platform,
  });
}

// ─── Router ────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    try {
      if (method === 'POST' && path === '/api/v1/register') {
        return await handleRegister(request, env);
      }
      if (method === 'POST' && path === '/api/v1/heartbeat') {
        return await handleHeartbeat(request, env);
      }
      if (method === 'GET' && path.startsWith('/api/v1/discover/')) {
        const email = path.slice('/api/v1/discover/'.length);
        return await handleDiscover(email, env);
      }

      // Health check
      if (path === '/health') {
        return json({ status: 'ok', service: 'hermes-bridge-registry' });
      }

      return json({ error: 'Not found' }, 404);
    } catch (e) {
      return json({ error: e.message }, 500);
    }
  },
};
