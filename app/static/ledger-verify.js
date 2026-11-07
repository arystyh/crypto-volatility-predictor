/* Verifikasi hash chain forecast ledger di browser / Node. HARUS identik dengan app/services/ledger.py
   (diuji di tests/test_explorer.py terhadap row_hash Python). */
(function (root) {
"use strict";
// --- harus identik dengan app/services/ledger.py ---
const GENESIS = "0".repeat(64);
const HASHED = ["symbol","decision_ts","horizon_min","model_version","prob","threshold","alerted","baseline_mu",
  "baseline_sd","z_threshold","logged_at","status","conformal_alpha","conformal_gamma"];
const FLOATS = new Set(["prob","threshold","baseline_mu","baseline_sd","z_threshold","conformal_alpha","conformal_gamma"]);

// repr(float) Python: digit terpendek yang round-trip (sama dengan JS), tapi notasi ilmiah bila exp < -4 atau >= 16.
function pyRepr(x){
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const neg = x < 0, s = Math.abs(x).toExponential();          // "d.ddde±X", digit terpendek
  const [mant, e] = s.split("e"); const exp = parseInt(e, 10);
  const digits = mant.replace(".", "");
  let out;
  if (exp < -4 || exp >= 16) {
    out = (digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits) + "e" + (exp < 0 ? "-" : "+") + String(Math.abs(exp)).padStart(2, "0");
  } else if (exp >= 0) {
    const intLen = exp + 1;
    const ip = digits.length > intLen ? digits.slice(0, intLen) : digits + "0".repeat(intLen - digits.length);
    const fp = digits.length > intLen ? digits.slice(intLen) : "0";
    out = ip + "." + fp;
  } else {
    out = "0." + "0".repeat(-exp - 1) + digits;
  }
  return (neg ? "-" : "") + out;
}
function canonical(row){
  // json.dumps(sort_keys=True, separators=(",", ":")); float -> string repr()
  const keys = [...HASHED].sort();
  return "{" + keys.map(k => {
    let v = row[k];
    if (v === undefined) v = null;
    if (v !== null && FLOATS.has(k)) v = pyRepr(Number(v));
    return JSON.stringify(k) + ":" + JSON.stringify(v);
  }).join(",") + "}";
}

// SHA-256: SubtleCrypto bila tersedia (HTTPS/localhost), fallback JS murni untuk HTTP biasa.
const enc = new TextEncoder();
function hex(buf){ return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2,"0")).join(""); }
const K = new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
function sha256js(bytes){
  const l = bytes.length, n = ((l + 9 + 63) >> 6) << 6, m = new Uint8Array(n);
  m.set(bytes); m[l] = 0x80; const dv = new DataView(m.buffer);
  dv.setUint32(n - 4, (l * 8) >>> 0); dv.setUint32(n - 8, Math.floor(l / 0x20000000));
  const H = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
  const W = new Uint32Array(64), r = (x, k) => (x >>> k) | (x << (32 - k));
  for (let o = 0; o < n; o += 64) {
    for (let i = 0; i < 16; i++) W[i] = dv.getUint32(o + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = r(W[i-15],7) ^ r(W[i-15],18) ^ (W[i-15] >>> 3), s1 = r(W[i-2],17) ^ r(W[i-2],19) ^ (W[i-2] >>> 10);
      W[i] = (W[i-16] + s0 + W[i-7] + s1) | 0;
    }
    let [a,b,c,d,e,f,g,h] = H;
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (r(e,6) ^ r(e,11) ^ r(e,25)) + ((e & f) ^ (~e & g)) + K[i] + W[i]) | 0;
      const t2 = ((r(a,2) ^ r(a,13) ^ r(a,22)) + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
  }
  const out = new DataView(new ArrayBuffer(32)); H.forEach((v, i) => out.setUint32(i * 4, v));
  return hex(out.buffer);
}
async function sha256(str){
  const b = enc.encode(str);
  if (root.crypto && root.crypto.subtle) return hex(await root.crypto.subtle.digest("SHA-256", b));
  return sha256js(b);
}
async function rowHash(prev, row){ return sha256(prev + canonical(row)); }
const api = {GENESIS, HASHED, pyRepr, canonical, sha256js, sha256, rowHash};
root.LedgerVerify = api;
if (typeof module !== "undefined") module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
