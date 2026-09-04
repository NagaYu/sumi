// Sumi in the browser — rule layer + ONNX token classifier, entirely client-side.
//
// The rule patterns, the JP numbering plan, the context words and the merge
// precedence are NOT written here: they are read from rules.json, which is
// generated from the Python definitions by scripts/export_rules_json.py. That is
// deliberate — hand-copying the regexes would let this demo and the library drift
// apart without anyone noticing.

const HUB_MODEL = 'NagaYu/sumi-ja-pii';

// ---------------------------------------------------------------- normalisation

// Mirrors sumi.types.normalize: NFKC, newline unification, dash unification.
// U+30FC (katakana long vowel) is deliberately NOT folded — doing so would turn
// coffee (ko-hi-) into a hyphenated string.
const DASHES = ['‐', '‑', '‒', '–', '—', '―',
                '−', '－'];

export function normalize(text) {
  let t = text.normalize('NFKC').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  for (const d of DASHES) t = t.split(d).join('-');
  return t;
}

// ------------------------------------------------------------------ rule layer

function digitsOf(s) { return (s.match(/\d/g) || []).join(''); }

function isValidJpPhone(s, plan) {
  const d = digitsOf(s);
  if (!d.startsWith('0')) return false;
  if (d.length !== 10 && d.length !== 11) return false;
  const p2 = d.slice(0, 2), p3 = d.slice(0, 3), p4 = d.slice(0, 4);
  if (plan.mobile.includes(p3) || plan.ip.includes(p3)) return d.length === 11;
  if (plan.tollfree4.includes(p4)) {
    if (p4 === '0120') return d.length === 10;
    if (p4 === '0800') return d.length === 11;
    return d.length === 10;
  }
  if (plan.area2.includes(p2)) return d.length === 10;
  if (plan.area3.includes(p3)) return d.length === 10;
  return d.length === 10;   // 4-digit area codes
}

const DATE_SHAPE = /^(19|20)\d{2}[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])$/;

export class RuleLayer {
  constructor(bundle) {
    this.bundle = bundle;
    this.window = bundle.context_window ?? 12;
    this.specs = bundle.specs.map(s => ({ ...s, re: new RegExp(s.pattern, 'gu') }));
  }

  detect(text) {
    const cands = [];
    for (const spec of this.specs) {
      spec.re.lastIndex = 0;
      let m;
      while ((m = spec.re.exec(text)) !== null) {
        if (m[0].length === 0) { spec.re.lastIndex++; continue; }
        const start = m.index, end = start + m[0].length, val = m[0];

        if (spec.validator === 'jp_phone' && !isValidJpPhone(val, this.bundle.phone_plan)) continue;
        if (spec.reject_date_shape && DATE_SHAPE.test(val.trim())) continue;

        // Negative context is checked on the LEFT only: labels such as the words
        // for "part number" and "order number" precede their value, and looking
        // rightwards would drop a legitimate phone number that merely happens to
        // be followed by one.
        const left = text.slice(Math.max(0, start - this.window), start);
        if (spec.negative_context.some(w => left.includes(w))) continue;

        const ctx = text.slice(Math.max(0, start - this.window),
                               Math.min(text.length, end + this.window));
        const hit = spec.context.find(w => ctx.includes(w)) ?? null;
        if (spec.require_context && hit === null) continue;

        cands.push({
          start, end, label: spec.label, text: val,
          score: Math.min(0.99, spec.confidence + (hit ? 0.15 : 0)),
          source: 'rule',
          meta: { rule_id: spec.rule_id, matched_context: hit, priority: spec.priority },
        });
      }
    }
    return resolveOverlaps(cands,
      s => [-(s.meta.priority ?? 0), -s.score, -(s.end - s.start), s.start]);
  }
}

function resolveOverlaps(spans, keyFn) {
  const overlaps = (a, b) => a.start < b.end && b.start < a.end;
  const ordered = [...spans].sort((a, b) => {
    const ka = keyFn(a), kb = keyFn(b);
    for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i]) return ka[i] - kb[i];
    return 0;
  });
  const chosen = [];
  for (const s of ordered) if (!chosen.some(c => overlaps(s, c))) chosen.push(s);
  return chosen.sort((a, b) => a.start - b.start || a.end - b.end);
}

// Mirrors sumi.rules.merge_spans — the five ordered steps, kept explicit.
export function mergeSpans(modelSpans, ruleSpans, ruleTypes) {
  const overlaps = (a, b) => a.start < b.end && b.start < a.end;
  const out = [];

  // 1. rule spans for the format-determined types are accepted unconditionally
  const accepted = [];
  for (const s of [...ruleSpans].sort((a, b) => a.start - b.start || a.end - b.end)) {
    if (!ruleTypes.includes(s.label)) continue;
    if (accepted.some(a => overlaps(s, a))) continue;
    accepted.push(s);
  }
  for (const s of accepted) {
    out.push({ ...s, source: 'merged', meta: { ...s.meta, from: 'rule' } });
  }

  // 2. model spans overlapping an accepted rule span are discarded
  const survivors = modelSpans.filter(m => !accepted.some(r => overlaps(m, r)));

  // 3-4. remaining model spans, highest score first, non-overlapping
  const kept = [];
  const byScore = [...survivors].sort(
    (a, b) => b.score - a.score || (b.end - b.start) - (a.end - a.start) || a.start - b.start);
  for (const m of byScore) if (!kept.some(k => overlaps(m, k))) kept.push(m);
  for (const m of kept) {
    out.push({ ...m, source: 'merged', meta: { ...m.meta, from: 'model' } });
  }

  // 5. sorted, non-overlapping
  return out.sort((a, b) => a.start - b.start || a.end - b.end);
}

// ------------------------------------------------------- boundary refinement

const HONORIFICS = ['様', 'さん', '氏', '殿', '君',
                    '先生', '部長', '課長', '社長',
                    'ちゃん'];
const PARTICLES = ['は', 'が', 'の', 'を', 'に', 'へ',
                   'と', 'より', 'から', 'で', 'も'];
const EDGE_PUNCT = /[\s　、。,.・:;]/;

// Mirrors sumi.model.refine_boundaries: strip trailing honorifics and particles
// from NAME and ADDRESS. Never returns an empty span.
export function refineBoundaries(span, text) {
  if (span.label !== 'NAME' && span.label !== 'ADDRESS') return span;
  let { start, end } = span;
  let changed = true;
  while (changed && end > start) {
    changed = false;
    const cur = text.slice(start, end);
    for (const h of [...HONORIFICS, ...PARTICLES]) {
      if (cur.length > h.length && cur.endsWith(h)) { end -= h.length; changed = true; break; }
    }
    const c = text.slice(start, end);
    if (c.length > 1 && EDGE_PUNCT.test(c.slice(-1))) { end -= 1; changed = true; }
    if (c.length > 1 && EDGE_PUNCT.test(c.slice(0, 1))) { start += 1; changed = true; }
  }
  if (end <= start) return span;
  return { ...span, start, end, text: text.slice(start, end) };
}

// --------------------------------------------------------------- model layer

// transformers.js does not expose offset mapping, so offsets are recovered by
// decoding each token and walking a cursor through the text. The ModernBERT-Ja
// tokenizer round-trips Japanese exactly, so a straight cursor scan is enough;
// anything that fails to match gets a zero-width span and is ignored downstream.
function alignOffsets(tokens, text) {
  const offsets = [];
  let cursor = 0;
  for (const raw of tokens) {
    if (!raw || /^<[^>]*>$/.test(raw)) { offsets.push([cursor, cursor]); continue; }
    const piece = raw.split('▁').join(' ');
    let idx = text.indexOf(piece, cursor);
    if (idx < 0) {
      const trimmed = piece.trim();
      idx = trimmed ? text.indexOf(trimmed, cursor) : -1;
      if (idx < 0) { offsets.push([cursor, cursor]); continue; }
      offsets.push([idx, idx + trimmed.length]);
      cursor = idx + trimmed.length;
      continue;
    }
    offsets.push([idx, idx + piece.length]);
    cursor = idx + piece.length;
  }
  return offsets;
}

function softmaxRow(arr) {
  const m = Math.max(...arr);
  const e = arr.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

// Mirrors sumi.model.decode_bio. A span's score is the MINIMUM over its
// constituent token probabilities — the weakest link, not the average.
function decodeBio(probs, offsets, text, labels, threshold) {
  const spans = [];
  let cur = null;
  const flush = () => {
    if (cur && cur.end > cur.start) {
      // Shrink away leading/trailing whitespace before emitting. The Python side
      // gets exact offsets from the tokenizer and never sees a whitespace-only
      // token; here offsets are reconstructed, so a metaspace token can otherwise
      // surface as a one-character span sitting on a blank.
      let { start, end } = cur;
      while (start < end && /\s/.test(text[start])) start++;
      while (end > start && /\s/.test(text[end - 1])) end--;
      if (end > start) {
        spans.push({
          start, end, label: cur.label, text: text.slice(start, end),
          score: cur.score, source: 'model', meta: {},
        });
      }
    }
    cur = null;
  };
  for (let i = 0; i < offsets.length; i++) {
    const [a, b] = offsets[i];
    if (b <= a) continue;
    if (!text.slice(a, b).trim()) { continue; }   // whitespace-only token
    const row = probs[i];
    let best = 0;
    for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k;
    const tag = labels[best], p = row[best];
    if (!tag || tag === 'O' || p < threshold) { flush(); continue; }
    const bi = tag.slice(0, 1), type = tag.slice(2);
    if (bi === 'B' || !cur || cur.label !== type) {
      flush();
      cur = { start: a, end: b, label: type, score: p };
    } else {
      cur.end = b;
      cur.score = Math.min(cur.score, p);   // weakest link
    }
  }
  flush();
  return spans;
}

export class SumiDetector {
  constructor(rules, tokenizer, model, labels) {
    this.rules = new RuleLayer(rules);
    this.ruleTypes = rules.rule_deterministic;
    this.types = rules.types;
    this.tokenizer = tokenizer;
    this.model = model;
    this.labels = labels;
    this.threshold = 0.5;
  }

  static async load(onProgress) {
    const [T, rules] = await Promise.all([
      import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6'),
      fetch('rules.json').then(r => r.json()),
    ]);
    T.env.allowLocalModels = false;
    const tokenizer = await T.AutoTokenizer.from_pretrained(HUB_MODEL);
    const model = await T.AutoModelForTokenClassification.from_pretrained(HUB_MODEL, {
      dtype: 'q8', progress_callback: onProgress,
    });
    const id2label = model.config.id2label || {};
    const labels = Object.keys(id2label)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => id2label[k]);
    return new SumiDetector(rules, tokenizer, model, labels);
  }

  async detect(rawText) {
    const text = normalize(rawText);

    const t0 = performance.now();
    const ruleSpans = this.rules.detect(text);
    const tRules = performance.now() - t0;

    const t1 = performance.now();
    const enc = this.tokenizer(text, { truncation: true, max_length: 512 });
    const ids = Array.from(enc.input_ids.data).map(Number);
    const tokens = ids.map(i => this.tokenizer.decode([i], { skip_special_tokens: false }));
    const offsets = alignOffsets(tokens, text);

    const out = await this.model({
      input_ids: enc.input_ids, attention_mask: enc.attention_mask,
    });
    const dims = out.logits.dims;
    const seq = dims[1], n = dims[2];
    const flat = out.logits.data;
    const probs = [];
    for (let i = 0; i < seq; i++) {
      const row = [];
      for (let k = 0; k < n; k++) row.push(Number(flat[i * n + k]));
      probs.push(softmaxRow(row));
    }
    const modelSpans = decodeBio(probs, offsets, text, this.labels, 0.05)
      .map(s => refineBoundaries(s, text))
      .filter(s => s.score >= this.threshold);
    const tModel = performance.now() - t1;

    const t2 = performance.now();
    const spans = mergeSpans(modelSpans, ruleSpans, this.ruleTypes);
    const tMerge = performance.now() - t2;

    return {
      text, spans,
      timings: { rules: tRules, model: tModel, merge: tMerge,
                 total: tRules + tModel + tMerge },
    };
  }
}

// ------------------------------------------------------------------- masking

// Mirrors sumi.mask.ReversibleMasker: stable placeholders, right-to-left
// substitution, and the same original value always maps to the same placeholder.
export function mask(text, spans) {
  const ordered = [];
  const bySpan = [...spans].sort(
    (a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
  for (const s of bySpan) {
    if (ordered.length && s.start < ordered[ordered.length - 1].end) continue;
    ordered.push(s);
  }
  const counters = {}, assigned = new Map(), entries = [];
  for (const s of ordered) {
    const original = text.slice(s.start, s.end);
    const key = original.normalize('NFKC') + ' ' + s.label;
    let ph = assigned.get(key);
    if (!ph) {
      counters[s.label] = (counters[s.label] || 0) + 1;
      ph = '<' + s.label + '_' + counters[s.label] + '>';
      assigned.set(key, ph);
    }
    entries.push({ placeholder: ph, original, label: s.label, start: s.start, end: s.end });
  }
  let out = text;
  for (const e of [...entries].sort((a, b) => b.start - a.start)) {
    out = out.slice(0, e.start) + e.placeholder + out.slice(e.end);
  }
  return { masked: out, entries };
}

export function unmask(text, entries) {
  const table = new Map(entries.map(e => [e.placeholder, e.original]));
  return text.replace(/<([A-Z_]+)_(\d+)>/g, m => (table.has(m) ? table.get(m) : m));
}
