//! Hy4-flash Engram n-gram hashing, FP8 alternate mapped-table reader
//! (`fp8_e4m3_f32scale_row_interleaved_v1`, 100-byte rows) with BF16 fallback,
//! and bitwise-exact BF16 decode/QDQ verification.
//!
//! Matches `python/reference/families/hy4_flash/model.py` (`engram_rows`) and
//! `quantization/hy4_flash/engram.py` (`decode_rows` / `_codes_and_scale`).
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::formats::mapped_table::row_cache::RowCache;
use crate::formats::mapped_table::{GatherMode, ReadPriority};
use crate::plan::checkpoint::mc::{EngramFp8Layer, ENGRAM_FP8_LAYOUT};
use crate::plan::checkpoint::Checkpoint;
use crate::{MappedTable, RowFormat, TablePart};

use super::Hy4FlashConfig;

/// Elements per Engram embedding row (`engram_n_embed_per_ngram / engram_n_head_per_ngram`).
pub const ENGRAM_EMBED_WIDTH: usize = 96;
/// Bytes per row in the FP8 alternate (`96` E4M3 codes + little-endian `F32` scale).
pub const ENGRAM_FP8_ROW_BYTES: usize = 100;
/// Bytes per row in the standard BF16 table (`96 * 2`).
pub const ENGRAM_BF16_ROW_BYTES: usize = ENGRAM_EMBED_WIDTH * 2;
const DEFAULT_CACHE_SHARES: [usize; 2] = [4096, 1024];

// ---------------------------------------------------------------------------
// Pure-Rust NumPy SeedSequence + PCG64 + Lemire bounded uint64
// ---------------------------------------------------------------------------

const SEED_SEQ_INIT_A: u32 = 0x43b0_d7e5;
const SEED_SEQ_MULT_A: u32 = 0x931e_8875;
const SEED_SEQ_INIT_B: u32 = 0x8b51_f9dd;
const SEED_SEQ_MULT_B: u32 = 0x58f3_8ded;
const SEED_SEQ_MIX_L: u32 = 0xca01_f9dd;
const SEED_SEQ_MIX_R: u32 = 0x4973_f715;
const PCG_MULT_128: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

fn seed_sequence_u64_words(seed: u64) -> [u64; 4] {
    let mut entropy = Vec::with_capacity(2);
    if seed == 0 {
        entropy.push(0u32);
    } else {
        let mut rem = seed;
        while rem > 0 {
            entropy.push(rem as u32);
            rem >>= 32;
        }
    }
    let mut hash_const = SEED_SEQ_INIT_A;
    let mut hashmix = |mut v: u32| -> u32 {
        v ^= hash_const;
        hash_const = hash_const.wrapping_mul(SEED_SEQ_MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = SEED_SEQ_MIX_L.wrapping_mul(x).wrapping_sub(SEED_SEQ_MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]));
            }
        }
    }
    let mut out_const = SEED_SEQ_INIT_B;
    let mut words32 = [0u32; 8];
    for (i, slot) in words32.iter_mut().enumerate() {
        let mut d = pool[i % 4] ^ out_const;
        out_const = out_const.wrapping_mul(SEED_SEQ_MULT_B);
        d = d.wrapping_mul(out_const);
        *slot = d ^ (d >> 16);
    }
    [
        u64::from(words32[0]) | (u64::from(words32[1]) << 32),
        u64::from(words32[2]) | (u64::from(words32[3]) << 32),
        u64::from(words32[4]) | (u64::from(words32[5]) << 32),
        u64::from(words32[6]) | (u64::from(words32[7]) << 32),
    ]
}

struct NumpyPcg64 {
    state: u128,
    inc: u128,
}

impl NumpyPcg64 {
    fn from_seed(seed: u64) -> Self {
        let words = seed_sequence_u64_words(seed);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let inc = (initseq << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT_128).wrapping_add(self.inc);
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step();
        let x = ((self.state >> 64) ^ self.state) as u64;
        let rot = (self.state >> 122) as u32;
        x.rotate_right(rot)
    }

    /// Exact match to NumPy's `bounded_lemire_uint64` (`rng.integers(0, bound, dtype=np.int64)`).
    fn next_bounded_u64(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let rng = bound - 1;
        let range = u128::from(rng) + 1;
        let mut m = u128::from(self.next_u64()) * range;
        let mut left = m as u64;
        if left < bound {
            let threshold = bound.wrapping_neg() % bound;
            while left < threshold {
                m = u128::from(self.next_u64()) * range;
                left = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// Deterministic Hy4-flash per-layer n-gram hasher matching `model.py:engram_rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4EngramHasher {
    pub layer: usize,
    pub lookup: Vec<i32>,
    pub factors: Vec<i64>,
    pub sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    pub pad_id: u32,
    pub masked_tokens: [u32; 3],
}

/// Streaming n-gram token history (`max_ngram_size - 1` previous tokens, padded with `engram_pad_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4EngramHistory(pub Vec<u32>);

impl Hy4EngramHasher {
    pub fn new(cfg: &Hy4FlashConfig, layer: usize, lookup: Vec<i32>) -> Result<Self> {
        ensure!(lookup.len() == cfg.engram_lookup_table_size,
            "Hy4 Engram layer {layer} lookup_table length {} != {}", lookup.len(), cfg.engram_lookup_table_size);
        let max_val = lookup.iter().copied().max().context("empty Hy4 Engram lookup_table")?;
        ensure!(max_val >= 0 && lookup.iter().all(|&v| v >= 0),
            "Hy4 Engram layer {layer} lookup_table contains negative compressed ids");
        ensure!((cfg.engram_pad_id as usize) < lookup.len(),
            "Hy4 Engram pad_id {} outside lookup_table {}", cfg.engram_pad_id, lookup.len());
        let sizes = cfg.engram_vocab_sizes.get(&layer)
            .with_context(|| format!("Hy4-flash layer {layer} has no Engram config"))?.clone();
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut total = 0u64;
        for &size in &sizes {
            offsets.push(total);
            total = total.checked_add(size).context("Hy4 Engram vocab size sum overflow")?;
        }
        let bound = (i64::MAX / (i64::from(max_val) + 1) / 2).max(1) as u64;
        let seed = cfg.engram_seed.checked_add(10_007u64 * (layer as u64 + 1))
            .context("Hy4 Engram seed overflow")?;
        let mut rng = NumpyPcg64::from_seed(seed);
        let factors = (0..cfg.engram_max_ngram_size)
            .map(|_| (rng.next_bounded_u64(bound) as i64) * 2 + 1)
            .collect();
        Ok(Self {
            layer,
            lookup,
            factors,
            sizes,
            offsets,
            heads_per_ngram: cfg.engram_n_head_per_ngram,
            pad_id: cfg.engram_pad_id,
            masked_tokens: cfg.masked_engram_tokens(),
        })
    }

    pub fn ngram_size(&self) -> usize { self.factors.len() }
    pub fn heads(&self) -> usize { self.sizes.len() }
    pub fn total_rows(&self) -> u64 { self.sizes.iter().sum() }

    pub fn start(&self) -> Hy4EngramHistory {
        Hy4EngramHistory(vec![self.pad_id; self.ngram_size() - 1])
    }

    #[inline]
    pub fn is_masked(&self, token: u32) -> bool {
        self.masked_tokens.contains(&token)
    }

    #[inline]
    fn compress(&self, token: u32) -> i64 {
        let idx = if (token as usize) < self.lookup.len() { token as usize } else { self.pad_id as usize };
        i64::from(self.lookup[idx])
    }

    /// Hash `tokens` starting from `history`, appending `tokens.len() * self.heads()`
    /// global table row IDs to `out` and updating `history` for streaming continuity.
    pub fn hash(&self, history: &mut Hy4EngramHistory, tokens: &[u32], out: &mut Vec<u64>) -> Result<()> {
        let ctx = self.ngram_size() - 1;
        ensure!(history.0.len() == ctx, "Hy4 Engram history must hold {ctx} tokens");
        if tokens.is_empty() { return Ok(()); }
        let mut window = Vec::with_capacity(ctx + tokens.len());
        window.extend_from_slice(&history.0);
        window.extend_from_slice(tokens);
        out.reserve(tokens.len() * self.heads());
        for pos in ctx..window.len() {
            let mut acc = self.compress(window[pos]) * self.factors[0];
            for order in 1..self.ngram_size() {
                acc ^= self.compress(window[pos - order]) * self.factors[order];
                let base = (order - 1) * self.heads_per_ngram;
                for h in base..base + self.heads_per_ngram {
                    let row = acc.rem_euclid(self.sizes[h] as i64) as u64 + self.offsets[h];
                    out.push(row);
                }
            }
        }
        history.0.copy_from_slice(&window[window.len() - ctx..]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bitwise-exact OCP E4M3FN <-> F32 / BF16 codec matching quantization/hy4_flash/engram.py
// ---------------------------------------------------------------------------

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// Exact OCP FP8 E4M3FN (`float8_e4m3fn`) byte to F32.
#[inline]
pub fn e4m3fn_to_f32(code: u8) -> f32 {
    let sign = u32::from(code & 0x80) << 24;
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = (f32::from(mant)) * f32::from_bits((127 - 9) << 23); // mant * 2^-9
        return f32::from_bits(val.to_bits() | sign);
    }
    if exp == 15 && mant == 7 {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    let f_exp = u32::from(exp) + (127 - 7);
    let f_mant = u32::from(mant) << 20;
    f32::from_bits(sign | (f_exp << 23) | f_mant)
}

/// Exact F32 to OCP FP8 E4M3FN (`float8_e4m3fn`) with round-to-nearest-even and saturation at ±448.
#[inline]
pub fn f32_to_e4m3fn_rne(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let abs_bits = bits & 0x7fff_ffff;
    if abs_bits > 0x7f80_0000 {
        return sign | 0x7f; // NaN
    }
    if abs_bits >= 0x43e0_0000 {
        // >= 448.0 saturates to 0x7e (448.0), matching torch clamp(-448, 448).to(float8_e4m3fn)
        return sign | 0x7e;
    }
    if abs_bits == 0 {
        return sign;
    }
    let f_exp = ((abs_bits >> 23) & 0xff) as i32 - 127;
    let f_mant = abs_bits & 0x007f_ffff;
    if f_exp < -6 {
        // Subnormal or underflow in E4M3FN (min positive subnormal is 2^-9; bias 7, 3 mant bits).
        let shift = (-6 - f_exp) as u32; // >= 1
        if shift > 24 {
            return sign;
        }
        let sig = (1u32 << 23) | f_mant;
        // We want round(sig / 2^(20 + shift))
        let total_shift = 20 + shift;
        if total_shift >= 32 {
            return sign;
        }
        let q = sig >> total_shift;
        let rem = sig & ((1u32 << total_shift) - 1);
        let half = 1u32 << (total_shift - 1);
        let rounded = if rem > half || (rem == half && (q & 1) == 1) { q + 1 } else { q };
        if rounded >= 8 {
            return sign | (1u8 << 3); // promoted to smallest normal (exp=1, mant=0)
        }
        return sign | (rounded as u8);
    }
    let e4 = (f_exp + 7) as u32;
    let q = f_mant >> 20;
    let rem = f_mant & 0x000f_ffff;
    let half = 0x0008_0000;
    let mut m3 = if rem > half || (rem == half && (q & 1) == 1) { q + 1 } else { q };
    let mut e = e4;
    if m3 >= 8 {
        m3 = 0;
        e += 1;
    }
    if e > 15 || (e == 15 && m3 >= 7) {
        return sign | 0x7e;
    }
    sign | ((e as u8) << 3) | (m3 as u8)
}

/// Decode one 100-byte `fp8_e4m3_f32scale_row_interleaved_v1` row into 96 BF16 elements:
/// `BF16(F32(code_e4m3) * scale_f32)`. Bitwise identical to `quantization/hy4_flash/engram.py:decode_rows`.
#[inline]
pub fn decode_fp8_row(packed: &[u8; ENGRAM_FP8_ROW_BYTES], out_bf16: &mut [u16; ENGRAM_EMBED_WIDTH]) {
    let scale = f32::from_le_bytes([packed[96], packed[97], packed[98], packed[99]]);
    for j in 0..ENGRAM_EMBED_WIDTH {
        out_bf16[j] = f32_to_bf16_rne(e4m3fn_to_f32(packed[j]) * scale);
    }
}

/// Exact stock Transformers BF16 Engram per-row E4M3 round-trip:
/// `x + (decoded - x)` in BF16 arithmetic, used for the BF16 table fallback path.
pub fn vendor_bf16_roundtrip(src_bf16: &[u16; ENGRAM_EMBED_WIDTH], out_bf16: &mut [u16; ENGRAM_EMBED_WIDTH]) {
    const FP8_MAX: f32 = 448.0;
    let floor = f32_to_bf16_rne(1e-12);
    let floor_f32 = bf16_to_f32(floor);
    let mut amax = 0.0f32;
    for &b in src_bf16 {
        let a = bf16_to_f32(b & 0x7fff);
        if a > amax { amax = a; }
    }
    let clamped_amax = if amax < floor_f32 { floor_f32 } else { amax };
    let scale = bf16_to_f32(f32_to_bf16_rne(clamped_amax / FP8_MAX));
    for j in 0..ENGRAM_EMBED_WIDTH {
        let x = bf16_to_f32(src_bf16[j]);
        let scaled = bf16_to_f32(f32_to_bf16_rne(x / scale)).clamp(-FP8_MAX, FP8_MAX);
        let code = f32_to_e4m3fn_rne(scaled);
        let code_bf16 = bf16_to_f32(f32_to_bf16_rne(e4m3fn_to_f32(code)));
        let decoded = bf16_to_f32(f32_to_bf16_rne(code_bf16 * scale));
        let diff = bf16_to_f32(f32_to_bf16_rne(decoded - x));
        out_bf16[j] = f32_to_bf16_rne(x + diff);
    }
}

// ---------------------------------------------------------------------------
// Mapped table reader: FP8 alternate default + BF16 fallback
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hy4EngramStorage {
    /// Derived 100-byte row-interleaved E4M3 + F32 scale shards declared in `cuteafd-mc.json` `engram_fp8`.
    Fp8Strided,
    /// Official 192-byte BF16 embedding table indexed in the standard view.
    Bf16Fallback,
}

impl Hy4EngramStorage {
    pub const fn row_bytes(self) -> usize {
        match self {
            Self::Fp8Strided => ENGRAM_FP8_ROW_BYTES,
            Self::Bf16Fallback => ENGRAM_BF16_ROW_BYTES,
        }
    }
}

/// One Hy4-flash Engram embedding table backed by the shared page-cache `MappedTable` + `RowCache`.
pub struct Hy4EngramTable {
    layer: usize,
    storage: Hy4EngramStorage,
    table: MappedTable,
    cache: Arc<RowCache>,
}

fn read_fp8_part_header(path: &Path, expected_tensor: &str, expected_row_start: u64, expected_rows: u64) -> Result<u64> {
    let mut file = File::open(path).with_context(|| format!("opening FP8 Engram part {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!((2..=64 * 1024 * 1024).contains(&header_len),
        "invalid safetensors header length {header_len} in {}", path.display());
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("decoding safetensors header in {}", path.display()))?;
    let meta = header.get("__metadata__").and_then(serde_json::Value::as_object)
        .with_context(|| format!("FP8 Engram part {} lacks __metadata__", path.display()))?;
    ensure!(meta.get("engram_layout").and_then(serde_json::Value::as_str) == Some(ENGRAM_FP8_LAYOUT),
        "FP8 Engram part {} has wrong engram_layout", path.display());
    ensure!(meta.get("row_start").and_then(serde_json::Value::as_str) == Some(expected_row_start.to_string().as_str()),
        "FP8 Engram part {} row_start mismatch (expected {expected_row_start})", path.display());
    let entry = header.get(expected_tensor).and_then(serde_json::Value::as_object)
        .with_context(|| format!("FP8 Engram part {} lacks tensor {expected_tensor}", path.display()))?;
    ensure!(entry.get("dtype").and_then(serde_json::Value::as_str) == Some("U8"),
        "FP8 Engram part {} tensor {expected_tensor} must be U8", path.display());
    let shape = entry.get("shape").and_then(serde_json::Value::as_array)
        .context("FP8 Engram part missing shape")?;
    ensure!(shape.len() == 2
        && shape[0].as_u64() == Some(expected_rows)
        && shape[1].as_u64() == Some(ENGRAM_FP8_ROW_BYTES as u64),
        "FP8 Engram part {} tensor {expected_tensor} shape mismatch (expected [{expected_rows}, {ENGRAM_FP8_ROW_BYTES}])",
        path.display());
    let offsets = entry.get("data_offsets").and_then(serde_json::Value::as_array)
        .context("FP8 Engram part missing data_offsets")?;
    let start = offsets.first().and_then(serde_json::Value::as_u64).context("bad data_offsets[0]")?;
    let end = offsets.get(1).and_then(serde_json::Value::as_u64).context("bad data_offsets[1]")?;
    let expected_bytes = expected_rows.checked_mul(ENGRAM_FP8_ROW_BYTES as u64)
        .context("FP8 Engram payload overflow")?;
    ensure!(end.checked_sub(start) == Some(expected_bytes),
        "FP8 Engram part {} payload extent mismatch", path.display());
    let data_offset = 8u64.checked_add(header_len).and_then(|v| v.checked_add(start))
        .context("FP8 Engram data offset overflow")?;
    ensure!(data_offset.checked_add(expected_bytes).is_some_and(|e| e <= file_len),
        "FP8 Engram part {} is truncated", path.display());
    Ok(data_offset)
}

impl Hy4EngramTable {
    pub fn layer(&self) -> usize { self.layer }
    pub fn storage(&self) -> Hy4EngramStorage { self.storage }
    pub fn rows(&self) -> u64 { self.table.rows() }
    pub fn row_bytes(&self) -> usize { self.table.row_bytes() }
    pub fn mapped(&self) -> &MappedTable { &self.table }
    pub fn cache(&self) -> &Arc<RowCache> { &self.cache }

    /// Open a Hy4-flash Engram layer from `checkpoint`: prefers the `engram_fp8`
    /// alternate declared in `cuteafd-mc.json` when present, falling back to the
    /// indexed BF16 table when absent.
    ///
    /// # Safety
    /// Checkpoint shard files must remain immutable while this table is mapped.
    pub unsafe fn open(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig, layer: usize) -> Result<Self> {
        let expected_rows = cfg.engram_rows(layer)
            .with_context(|| format!("Hy4-flash layer {layer} is not an Engram layer"))?;
        if let Some(engram_fp8) = checkpoint.view.as_ref().and_then(|v| v.engram_fp8()) {
            let entry = engram_fp8.layer(layer)
                .with_context(|| format!("cuteafd-mc.json engram_fp8 lacks layer {layer}"))?;
            // SAFETY: forwarded from caller.
            return unsafe { Self::open_fp8_parts(&checkpoint.snapshot, layer, entry, expected_rows) };
        }
        // SAFETY: forwarded from caller.
        unsafe { Self::open_bf16(checkpoint, layer, expected_rows) }
    }

    /// Open strictly the `engram_fp8` strided parts for `layer`.
    ///
    /// # Safety
    /// Referenced shard files must remain immutable while mapped.
    pub unsafe fn open_fp8_parts(snapshot: &Path, layer: usize, entry: &EngramFp8Layer, expected_rows: u64) -> Result<Self> {
        let total = entry.total_rows()?;
        ensure!(total == expected_rows,
            "Hy4 Engram FP8 layer {layer} has {total} rows, config expects {expected_rows}");
        let mut parts = Vec::with_capacity(entry.parts.len());
        let mut row_start = 0u64;
        for part in &entry.parts {
            ensure!(!part.file.is_empty()
                && Path::new(&part.file).components().all(|c| matches!(c, Component::Normal(_))),
                "invalid FP8 Engram part shard path {}", part.file);
            let path = snapshot.join(&part.file);
            let offset = read_fp8_part_header(&path, &part.tensor, row_start, part.rows)?;
            parts.push(TablePart { path, offset, rows: part.rows });
            row_start = row_start.checked_add(part.rows).context("FP8 Engram row_start overflow")?;
        }
        // SAFETY: forwarded from caller.
        let table = unsafe { MappedTable::open(&parts, RowFormat::of(DType::U8, ENGRAM_FP8_ROW_BYTES)?)? };
        table.name_stats(format!("hy4_engram.{layer}.fp8_strided"));
        let cache = RowCache::new(ENGRAM_FP8_ROW_BYTES, DEFAULT_CACHE_SHARES, table.shared_stats())?;
        Ok(Self { layer, storage: Hy4EngramStorage::Fp8Strided, table, cache })
    }

    /// Open strictly the indexed BF16 Engram table for `layer` (fallback or verification source).
    ///
    /// # Safety
    /// Referenced shard file must remain immutable while mapped.
    pub unsafe fn open_bf16(checkpoint: &Checkpoint, layer: usize, expected_rows: u64) -> Result<Self> {
        let name = format!("model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight");
        let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("missing BF16 Engram tensor {name}"))?;
        ensure!(tensor.meta.dtype == DType::Bf16
            && tensor.meta.shape == [expected_rows as usize, ENGRAM_EMBED_WIDTH]
            && tensor.meta.byte_length == expected_rows * ENGRAM_BF16_ROW_BYTES as u64,
            "invalid BF16 Engram header for {name}: {:?} {:?}", tensor.meta.dtype, tensor.meta.shape);
        ensure!(!tensor.shard.is_empty()
            && Path::new(&tensor.shard).components().all(|c| matches!(c, Component::Normal(_))),
            "invalid BF16 Engram shard path {}", tensor.shard);
        let path = checkpoint.snapshot.join(&tensor.shard);
        // SAFETY: forwarded from caller.
        let table = unsafe {
            MappedTable::single(&path, tensor.meta.byte_offset, expected_rows, RowFormat::of(DType::Bf16, ENGRAM_EMBED_WIDTH)?)?
        };
        table.name_stats(format!("hy4_engram.{layer}.bf16"));
        let cache = RowCache::new(ENGRAM_BF16_ROW_BYTES, DEFAULT_CACHE_SHARES, table.shared_stats())?;
        Ok(Self { layer, storage: Hy4EngramStorage::Bf16Fallback, table, cache })
    }

    /// Gather raw row bytes via the shared mapped-table reactor into `out`.
    pub fn gather_raw(&self, rows: &[u64], priority: ReadPriority) -> Result<Vec<u8>> {
        let width = self.table.row_bytes();
        let mut out = vec![0u8; rows.len() * width];
        if rows.is_empty() { return Ok(out); }
        let slots: Vec<usize> = (0..rows.len()).collect();
        let mut prepared = self.table.prepare_gather(rows, &slots, &mut out, priority, 2, GatherMode::default())?;
        while !prepared.step(&self.table, &mut out)? {}
        Ok(out)
    }

    /// Gather `rows` and decode into BF16 (`rows.len() * 96` BF16 values),
    /// matching the stock BF16 lookup expression bitwise for both `Fp8Strided` and `Bf16Fallback`.
    pub fn gather_decoded_bf16(&self, rows: &[u64], priority: ReadPriority) -> Result<Vec<u16>> {
        let raw = self.gather_raw(rows, priority)?;
        let mut out = vec![0u16; rows.len() * ENGRAM_EMBED_WIDTH];
        match self.storage {
            Hy4EngramStorage::Fp8Strided => {
                for (i, chunk) in raw.chunks_exact(ENGRAM_FP8_ROW_BYTES).enumerate() {
                    let src: &[u8; ENGRAM_FP8_ROW_BYTES] = chunk.try_into().unwrap();
                    let dst: &mut [u16; ENGRAM_EMBED_WIDTH] =
                        (&mut out[i * ENGRAM_EMBED_WIDTH..(i + 1) * ENGRAM_EMBED_WIDTH]).try_into().unwrap();
                    decode_fp8_row(src, dst);
                }
            }
            Hy4EngramStorage::Bf16Fallback => {
                for (i, chunk) in raw.chunks_exact(ENGRAM_BF16_ROW_BYTES).enumerate() {
                    let mut src_bf16 = [0u16; ENGRAM_EMBED_WIDTH];
                    for (j, pair) in chunk.chunks_exact(2).enumerate() {
                        src_bf16[j] = u16::from_le_bytes([pair[0], pair[1]]);
                    }
                    let dst: &mut [u16; ENGRAM_EMBED_WIDTH] =
                        (&mut out[i * ENGRAM_EMBED_WIDTH..(i + 1) * ENGRAM_EMBED_WIDTH]).try_into().unwrap();
                    vendor_bf16_roundtrip(&src_bf16, dst);
                }
            }
        }
        Ok(out)
    }
}

/// Read the small per-layer `lookup_table` (I32 `[120818]`) directly from its shard.
pub fn read_lookup_table(checkpoint: &Checkpoint, layer: usize, expected_len: usize) -> Result<Vec<i32>> {
    let name = format!("model.layers.{layer}.engram_layer.lookup_table");
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .with_context(|| format!("missing Hy4 Engram tensor {name}"))?;
    ensure!(tensor.meta.dtype == DType::I32 && tensor.meta.shape == [expected_len]
        && tensor.meta.byte_length == expected_len as u64 * 4,
        "invalid Hy4 Engram lookup_table header for {name}: {:?} {:?}", tensor.meta.dtype, tensor.meta.shape);
    let mut file = File::open(checkpoint.snapshot.join(&tensor.shard))?;
    file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
    let mut bytes = vec![0u8; expected_len * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Total on-disk Engram table bytes across all declared layers for the active storage mode.
pub fn mapped_engram_bytes(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig) -> Result<(Hy4EngramStorage, u64, BTreeMap<usize, u64>)> {
    let storage = if checkpoint.view.as_ref().and_then(|v| v.engram_fp8()).is_some() {
        Hy4EngramStorage::Fp8Strided
    } else {
        Hy4EngramStorage::Bf16Fallback
    };
    let mut per_layer = BTreeMap::new();
    let mut total = 0u64;
    for &layer in &cfg.engram_layers {
        let rows = cfg.engram_rows(layer).context("missing engram rows")?;
        let bytes = rows.checked_mul(storage.row_bytes() as u64).context("engram bytes overflow")?;
        per_layer.insert(layer, bytes);
        total = total.checked_add(bytes).context("total engram bytes overflow")?;
    }
    Ok((storage, total, per_layer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::checkpoint::{CheckpointView, CoreView};

    #[test]
    fn numpy_pcg64_factors_match_reference_exactly() {
        // Exact values verified against `numpy.random.default_rng(0 + 10007 * (layer + 1))`
        // with `bound = i64::MAX // (92619 + 1) // 2 = 49_791_470_723_681`.
        let bound = (i64::MAX / 92_620 / 2) as u64;
        assert_eq!(bound, 49_791_470_723_681);
        for (layer, expected) in [
            (1usize, [81_853_575_799_317i64, 39_690_085_502_883, 24_229_915_018_165]),
            (11, [19_457_948_045_719, 44_729_556_510_419, 84_779_352_922_315]),
            (21, [72_974_230_057_307, 21_530_602_121_221, 55_998_057_243_097]),
            (31, [29_128_509_562_787, 16_745_155_128_523, 22_474_588_302_733]),
        ] {
            let mut rng = NumpyPcg64::from_seed(10_007u64 * (layer as u64 + 1));
            let got = [
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
            ];
            assert_eq!(got, expected, "layer {layer}");
        }
    }

    #[test]
    fn fp8_decode_matches_vendor_bf16_roundtrip_on_edges() {
        // Synthetic rows covering ±0, subnormals, 1e-12 floor, ±448 endpoints and midpoints.
        let mut row = [0u16; ENGRAM_EMBED_WIDTH];
        let samples = [
            0.0f32, -0.0, 1e-14, -1e-14, 1e-12, -1e-12, 0.001953125, -0.001953125,
            0.015625, 0.125, 0.375, 1.0, -1.5, 3.25, 7.5, -15.5, 128.0, -448.0, 448.0,
        ];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = f32_to_bf16_rne(samples[j % samples.len()]);
        }
        let mut reference = [0u16; ENGRAM_EMBED_WIDTH];
        vendor_bf16_roundtrip(&row, &mut reference);

        // Pack exactly as `quantization/hy4_flash/engram.py:_checked_pack` does.
        let floor = bf16_to_f32(f32_to_bf16_rne(1e-12));
        let amax = row.iter().map(|&b| bf16_to_f32(b & 0x7fff)).fold(0.0f32, f32::max).max(floor);
        let scale = bf16_to_f32(f32_to_bf16_rne(amax / 448.0));
        let mut packed = [0u8; ENGRAM_FP8_ROW_BYTES];
        for j in 0..ENGRAM_EMBED_WIDTH {
            let scaled = bf16_to_f32(f32_to_bf16_rne(bf16_to_f32(row[j]) / scale)).clamp(-448.0, 448.0);
            let mut code = f32_to_e4m3fn_rne(scaled);
            if (code & 0x7f) == 0 {
                code = if (reference[j] & 0x8000) != 0 { 0x80 } else { 0x00 };
            }
            packed[j] = code;
        }
        packed[96..100].copy_from_slice(&scale.to_le_bytes());

        let mut decoded = [0u16; ENGRAM_EMBED_WIDTH];
        decode_fp8_row(&packed, &mut decoded);
        assert_eq!(decoded, reference);
    }

    #[test]
    #[ignore = "installed Hy4-flash K4 MC bitwise FP8 vs BF16 Engram verification"]
    fn installed_hy4_engram_fp8_matches_bf16_bitwise_on_real_lookups() -> Result<()> {
        let home = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME").context("installed HF home required")?;
        let snapshot = Path::new(&home)
            .join("hub/models--wrldsuksgo2mars--Hy4-flash-EXL3-K4-MC/snapshots/cc72179113573d751ed3245e3e58d3640694378d");
        let view = CheckpointView::select(&snapshot, CoreView::Original)?;
        let checkpoint = Checkpoint::open_view(&view)?;
        let cfg = Hy4FlashConfig::from_hf(&checkpoint.config)?;
        let tokens = [120_000u32, 42, 100, 120_820, 7, 120_002, 500, 1024];

        for &layer in &cfg.engram_layers {
            let lookup = read_lookup_table(&checkpoint, layer, cfg.engram_lookup_table_size)?;
            let hasher = Hy4EngramHasher::new(&cfg, layer, lookup)?;
            let mut full_hist = hasher.start();
            let mut rows = Vec::new();
            hasher.hash(&mut full_hist, &tokens, &mut rows)?;

            // Streaming chunk-by-chunk must equal single pass.
            let mut split_hist = hasher.start();
            let mut split_rows = Vec::new();
            hasher.hash(&mut split_hist, &tokens[..3], &mut split_rows)?;
            hasher.hash(&mut split_hist, &tokens[3..], &mut split_rows)?;
            assert_eq!(rows, split_rows, "streaming mismatch layer {layer}");
            assert_eq!(full_hist, split_hist);

            if layer == 1 {
                // Exact row match against `python/reference/families/hy4_flash/model.py:engram_rows`.
                assert_eq!(&rows[0..16], &[
                    4_258_903, 16_210_259, 27_260_810, 31_206_235, 46_090_317, 56_488_174, 70_058_412, 75_384_515,
                    91_716_212, 103_088_154, 112_241_750, 118_054_132, 125_718_195, 134_064_584, 150_441_785, 156_008_176,
                ]);
                assert_eq!(&rows[48..64], &[
                    235_426, 18_785_785, 27_351_194, 33_367_523, 42_293_259, 52_181_829, 67_674_724, 81_119_641,
                    87_193_657, 102_541_383, 103_437_646, 117_820_561, 129_480_352, 138_758_752, 145_934_975, 159_886_767,
                ]);
            }

            let expected_rows = cfg.engram_rows(layer).unwrap();
            // SAFETY: immutable published checkpoint snapshot.
            let fp8_table = unsafe { Hy4EngramTable::open(&checkpoint, &cfg, layer)? };
            assert_eq!(fp8_table.storage(), Hy4EngramStorage::Fp8Strided);
            // SAFETY: immutable published checkpoint snapshot.
            let bf16_table = unsafe { Hy4EngramTable::open_bf16(&checkpoint, layer, expected_rows)? };
            assert_eq!(bf16_table.storage(), Hy4EngramStorage::Bf16Fallback);

            let fp8_decoded = fp8_table.gather_decoded_bf16(&rows, ReadPriority::Prefill)?;
            let bf16_decoded = bf16_table.gather_decoded_bf16(&rows, ReadPriority::Prefill)?;
            assert_eq!(fp8_decoded, bf16_decoded, "layer {layer} FP8 vs BF16 bit mismatch");
        }
        Ok(())
    }
}
