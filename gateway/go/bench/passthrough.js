// SPDX-License-Identifier: Apache-2.0
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — k6 passthrough load test
//
// Run: k6 run gateway/go/bench/passthrough.js
//
// Env:
//   DIRECTOR_URL      e.g. http://localhost:8080
//   DIRECTOR_KEY      matches one DIRECTOR_API_KEYS entry
//   DIRECTOR_VUS      virtual users (default 50)
//   DIRECTOR_DURATION e.g. 30s (default)

import http from 'k6/http';
import { check } from 'k6';

/** Base address of the gateway selected by the operator. */
const baseURL = __ENV.DIRECTOR_URL || 'http://localhost:8080';
/** Optional gateway API key supplied to the real HTTP request. */
const apiKey = __ENV.DIRECTOR_KEY || '';
/** Concurrent users requested by the operator. */
const vus = Number(__ENV.DIRECTOR_VUS || 50);
/** Finite load duration in the native k6 duration format. */
const duration = __ENV.DIRECTOR_DURATION || '30s';

/** Native k6 schedule and existing failure/latency thresholds. */
export const options = {
  vus: vus,
  duration: duration,
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<500', 'p(99)<1500'],
  },
};

/** Send one chat-completion request through the configured gateway. */
export default function () {
  const payload = JSON.stringify({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'user', content: 'What is 2+2?' },
    ],
    stream: false,
  });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    },
  };
  const resp = http.post(`${baseURL}/v1/chat/completions`, payload, params);
  check(resp, {
    'status is not 5xx': (r) => r.status < 500,
  });
}
