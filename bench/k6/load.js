import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE_URL || 'http://localhost:8000';
const QUESTIONS = [
  'What is retrieval-augmented generation?',
  'How do transformers handle long context?',
  'What are common methods for evaluating language models?',
];

export const options = {
  scenarios: {
    health: { executor: 'ramping-vus', exec: 'health', startVUs: 0,
      stages: [{ duration: '10s', target: 50 }, { duration: '20s', target: 50 }, { duration: '5s', target: 0 }] },
    page: { executor: 'constant-vus', exec: 'page', vus: 10, duration: '35s' },
    // 8 answers stay under the 10/hour/IP limit; real providers, real latency.
    answer: { executor: 'per-vu-iterations', exec: 'answer', vus: 1, iterations: 8, maxDuration: '5m' },
    // 15 more from a second client shows the limiter, not the model.
    burst: { executor: 'per-vu-iterations', exec: 'burst', vus: 1, iterations: 15, startTime: '40s' },
  },
  thresholds: {
    'http_req_duration{scenario:health}': ['p(95)<200'],
    'http_req_failed{scenario:health}': ['rate<0.01'],
    'http_req_duration{scenario:answer}': ['p(95)<40000'],
  },
};

export function health() { check(http.get(`${BASE}/health`), { ok: (r) => r.status === 200 }); }
export function page() { check(http.get(`${BASE}/`), { ok: (r) => r.status === 200 }); }
export function answer() {
  const q = QUESTIONS[__ITER % QUESTIONS.length] + ` (${__ITER})`;
  const r = http.post(`${BASE}/answer`, JSON.stringify({ question: q }),
    { headers: { 'Content-Type': 'application/json', 'CF-Connecting-IP': '10.0.0.1' }, timeout: '60s' });
  check(r, { 'answered 200': (x) => x.status === 200 });
}
export function burst() {
  const r = http.post(`${BASE}/answer`, JSON.stringify({ question: 'What is attention?' }),
    { headers: { 'Content-Type': 'application/json', 'CF-Connecting-IP': '10.0.0.2' } });
  check(r, { 'limited 429 or ok': (x) => [200, 429, 503].includes(x.status) });
}
