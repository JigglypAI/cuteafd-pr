//! Checked launch contracts for the native MiniMax residual/router/KV producers.
use super::{bytes, invalid, kv_plane_record_bytes, validate_buffers, Result};
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind { ResidualNorm, RouterScores, KvPack, KvPackHeads }
impl Kind {
    pub const ALL: [Self; 3] = [Self::ResidualNorm, Self::RouterScores, Self::KvPack];
    pub const HEADS: [Self; 3] = [Self::ResidualNorm, Self::RouterScores, Self::KvPackHeads];
    fn kv_heads(self) -> Option<u32> {
        match self { Self::KvPack => Some(4), Self::KvPackHeads => Some(2), _ => None }
    }
    fn op(self) -> &'static str {
        if self == Self::KvPackHeads { "minimax_kv_pack" } else { self.name() }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::ResidualNorm => "minimax_residual_norm",
            Self::RouterScores => "minimax_router_scores",
            Self::KvPack => "minimax_kv_pack",
            Self::KvPackHeads => "minimax_kv_pack_h2",
        }
    }
    fn names(self) -> &'static [&'static str] {
        match self {
            Self::ResidualNorm => &["x", "weight", "bias", "cos", "sin", "positions", "out"],
            Self::RouterScores => &["x", "weight", "logits"],
            Self::KvPack | Self::KvPackHeads => &["k", "v", "slots", "key_cache", "value_cache"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardLayout { pub kind: Kind, pub rows: u32 }
impl ForwardLayout {
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let kind = Kind::ALL.into_iter().chain([Kind::KvPackHeads]).find(|k| spec.name == k.name())
            .ok_or_else(|| invalid("unknown forward producer"))?;
        let g = &spec.geometry;
        let geometry = match kind {
            Kind::ResidualNorm => g["width"] == 6144 && g["heads"] == 1
                && g["eps"].as_f64() == Some(1e-6) && g["zero_centered"] == true
                && g["subtract_mean"] == false && g["bias"] == false && g["rotary"] == 0,
            Kind::RouterScores => g["hidden"] == 6144 && g["experts"] == 128,
            Kind::KvPack | Kind::KvPackHeads => g["heads"] == kind.kv_heads().unwrap() && g["head_dim"] == 128 && g["page_rows"] == 64
                && g["record_bytes"] == kv_plane_record_bytes(kind.kv_heads().unwrap() as usize)
                && g["scale_offset"] == kind.kv_heads().unwrap() * 128 * 64
                && g["scale_stride"] == kv_plane_record_bytes(kind.kv_heads().unwrap() as usize) - kind.kv_heads().unwrap() as usize * 128,
        };
        if !geometry || spec.capacity_rows == 0 || spec.capacity_rows > 4096
            || spec.scalar_kinds != b"i" || !spec.scratch.is_empty()
            || spec.pointers.iter().map(String::as_str).ne(kind.names().iter().copied()) {
            return Err(invalid("forward producer geometry, capacity or ABI"));
        }
        Ok(Self { kind, rows: spec.capacity_rows })
    }
    fn validate<const N: usize>(self, rows: usize, items: [(Buffer, usize); N],
        align: [usize; N], writable: &[usize]) -> Result<[Buffer; N]> {
        if rows == 0 || rows > self.rows as usize { return Err(invalid("forward producer live rows")); }
        validate_buffers(&items, &align, writable).map_err(invalid)?;
        Ok(items.map(|(buffer, _)| buffer))
    }
    fn norm(self, b: NormBatch) -> Result<[Buffer; 7]> {
        if self.kind != Kind::ResidualNorm { return Err(invalid("not a residual norm")); }
        // The shared affine norm ABI retains unused operands. Reuse the weight
        // for them: the specialization never reads bias/tables/positions.
        self.validate(b.rows, [(b.input, bytes(b.rows, 6144 * 4)?), (b.weight, 6144 * 2),
            (b.weight, 0), (b.weight, 0), (b.weight, 0), (b.weight, 0),
            (b.output, bytes(b.rows, 6144 * 2)?)], [16; 7], &[6])
    }
    fn router(self, b: RouterBatch) -> Result<[Buffer; 3]> {
        if self.kind != Kind::RouterScores { return Err(invalid("not router scores")); }
        self.validate(b.rows, [(b.input, bytes(b.rows, 6144 * 2)?), (b.weight, 128 * 6144 * 4),
            (b.output, bytes(b.rows, 128 * 4)?)], [16; 3], &[2])
    }
    fn kv(self, b: KvBatch) -> Result<[Buffer; 5]> {
        let h = self.kind.kv_heads().ok_or_else(|| invalid("not a KV producer"))? as usize;
        if b.pool_pages == 0 {
            return Err(invalid("KV producer geometry"));
        }
        self.validate(b.rows, [(b.key, bytes(b.rows, h * 256)?), (b.value, bytes(b.rows, h * 256)?),
            (b.slots, bytes(b.rows, 8)?),
            (b.key_cache, bytes(b.pool_pages, 64 * kv_plane_record_bytes(h))?),
            (b.value_cache, bytes(b.pool_pages, 64 * kv_plane_record_bytes(h))?)], [16; 5], &[3, 4])
    }
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<ForwardLayout> {
    let layout = ForwardLayout::from_spec(spec)?;
    let h = layout.kind.kv_heads().unwrap_or(0);
    let kv = [("bfloat16", format!("[rows,{h},128]"), "in"),
        ("bfloat16", format!("[rows,{h},128]"), "in"),
        ("int64", "[rows] (physical slots; negative skips writes)".into(), "in"),
        ("uint8", format!("[pages,64,{}] scaled E4M3 records", kv_plane_record_bytes(h as usize)), "out"),
        ("uint8", format!("[pages,64,{}] scaled E4M3 records", kv_plane_record_bytes(h as usize)), "out")];
    let kv_refs = kv.iter().map(|(d, s, r)| (*d, s.as_str(), *r)).collect::<Vec<_>>();
    let operands: &[(&str, &str, &str)] = match layout.kind {
        Kind::ResidualNorm => &[("float32", "[rows,1,6144]", "in"), ("bfloat16", "[6144]", "in"),
            ("bfloat16", "[6144] (ignored unless bias)", "in"),
            ("bfloat16", "[positions,0] (ignored unless rotary)", "in"),
            ("bfloat16", "[positions,0] (ignored unless rotary)", "in"),
            ("int64", "[rows] (ignored unless rotary)", "in"), ("bfloat16", "[rows,1,6144]", "out")],
        Kind::RouterScores => &[("bfloat16", "[rows,6144]", "in"),
            ("float32", "[128,6144]", "in"), ("float32", "[rows,128]", "out")],
        Kind::KvPack | Kind::KvPackHeads => &kv_refs,
    };
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("forward pointer metadata"))?;
    if entry["op"] != layout.kind.op() || pointers.len() != operands.len()
        || pointers.iter().zip(layout.kind.names().iter().zip(operands))
            .any(|(p, (name, (dtype, shape, role)))| p["name"] != *name || p["dtype"] != *dtype
                || p["shape"] != *shape || p["role"] != *role)
        || entry["scalars"] != serde_json::json!([{"name":"rows","type":"int32"}]) {
        return Err(invalid("forward exported operand or operation contract"));
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
pub struct NormBatch { pub rows: usize, pub input: Buffer, pub weight: Buffer, pub output: Buffer }
#[derive(Clone, Copy)]
pub struct RouterBatch { pub rows: usize, pub input: Buffer, pub weight: Buffer, pub output: Buffer }
#[derive(Clone, Copy)]
pub struct KvBatch {
    pub rows: usize,
    pub pool_pages: usize,
    pub key: Buffer,
    pub value: Buffer,
    pub slots: Buffer,
    pub key_cache: Buffer,
    pub value_cache: Buffer,
}

pub struct ForwardProducer<'a> { program: Program<'a>, pub layout: ForwardLayout }
impl<'a> ForwardProducer<'a> {
    pub fn bind(programs: &'a Programs<'a>, kind: Kind) -> Result<Self> {
        let spec = programs.spec(kind.name()).map_err(|e| invalid(e.to_string()))?;
        let layout = ForwardLayout::from_spec(spec)?;
        let program = programs.program(kind.name(), kind.names()).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }
    /// # Safety
    /// Storage and stream must remain live through queued work and graph use.
    pub unsafe fn norm(&self, batch: NormBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.norm(batch)?;
        // SAFETY: host extents/devices/aliases checked; caller retains storage.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Storage and stream must remain live through queued work and graph use.
    pub unsafe fn router(&self, batch: RouterBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.router(batch)?;
        // SAFETY: host extents/devices/aliases checked; caller retains storage.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
    /// # Safety
    /// Nonnegative slots must be unique and address the scaled cache. Retain all storage
    /// and stream through queued work and graph use. Negative slots skip writes.
    pub unsafe fn kv(&self, batch: KvBatch, stream: *mut c_void) -> Result<()> {
        let buffers = self.layout.kv(batch)?;
        // SAFETY: host extents/devices/aliases checked; device indices and lifetimes are caller-owned.
        unsafe { self.program.launch(&buffers.map(|b| b.ptr), &[Scalar::I32(batch.rows as i32)], stream) }
            .map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn buffer(i: usize) -> Buffer {
        Buffer { ptr: ((i + 1) * 0x100000000usize) as *mut c_void, bytes: 1 << 30, device_id: 0, flags: 0 }
    }
    fn spec(kind: Kind) -> ProgramSpec {
        let geometry = match kind {
            Kind::ResidualNorm => serde_json::json!({"width":6144,"heads":1,"eps":1e-6,
                "zero_centered":true,"subtract_mean":false,"bias":false,"rotary":0}),
            Kind::RouterScores => serde_json::json!({"hidden":6144,"experts":128}),
            Kind::KvPack | Kind::KvPackHeads => serde_json::json!({"heads":kind.kv_heads().unwrap(),"head_dim":128,"page_rows":64,
                "record_bytes":kv_plane_record_bytes(kind.kv_heads().unwrap() as usize),"scale_offset":kind.kv_heads().unwrap()*128*64,"scale_stride":kv_plane_record_bytes(kind.kv_heads().unwrap() as usize)-kind.kv_heads().unwrap() as usize*128}),
        };
        ProgramSpec { name: kind.name().into(), pointers: kind.names().iter().map(|s| (*s).into()).collect(),
            scalar_kinds: b"i".to_vec(), scratch: Default::default(), capacity_rows: 4096, geometry }
    }
    #[test]
    fn accepts_exact_specs_rejects_wrong_abi_geometry_and_capacity() {
        for kind in Kind::ALL {
            let expected = spec(kind);
            assert_eq!(ForwardLayout::from_spec(&expected).unwrap(), ForwardLayout { kind, rows: 4096 });
            let mut wrong = expected.clone(); wrong.pointers.swap(0, 1);
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected.clone(); wrong.scalar_kinds = b"q".to_vec();
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected.clone(); wrong.capacity_rows = 4097;
            assert!(ForwardLayout::from_spec(&wrong).is_err());
            let mut wrong = expected; wrong.geometry = serde_json::json!({});
            assert!(ForwardLayout::from_spec(&wrong).is_err());
        }
    }
    #[test]
    fn validates_forward_manifest_operands_before_loading() {
        for kind in Kind::ALL {
            let spec = spec(kind);
            // The exporter ABI is independently pinned by the fork's GPU oracles.
            let operands = match kind {
                Kind::ResidualNorm => vec![("float32", "[rows,1,6144]"), ("bfloat16", "[6144]"),
                    ("bfloat16", "[6144] (ignored unless bias)"),
                    ("bfloat16", "[positions,0] (ignored unless rotary)"),
                    ("bfloat16", "[positions,0] (ignored unless rotary)"),
                    ("int64", "[rows] (ignored unless rotary)"), ("bfloat16", "[rows,1,6144]")],
                Kind::RouterScores => vec![("bfloat16", "[rows,6144]"),
                    ("float32", "[128,6144]"), ("float32", "[rows,128]")],
                Kind::KvPack => vec![("bfloat16", "[rows,4,128]"), ("bfloat16", "[rows,4,128]"),
                    ("int64", "[rows] (physical slots; negative skips writes)"),
                    ("uint8", "[pages,64,528] scaled E4M3 records"), ("uint8", "[pages,64,528] scaled E4M3 records")],
                Kind::KvPackHeads => unreachable!("whole fixture"),
            };
            let entry = serde_json::json!({"op":kind.name(),
                "scalars":[{"name":"rows","type":"int32"}],
                "pointers":kind.names().iter().zip(operands).enumerate().map(|(i,(n,(d,s)))|
                    serde_json::json!({"name":n,"dtype":d,"shape":s,
                        "role":if i + 1 == kind.names().len() || kind == Kind::KvPack && i == 3 {"out"} else {"in"}}))
                    .collect::<Vec<_>>()});
            assert!(validate_entry(&spec, &entry).is_ok());
            let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests().unwrap();
            let library = fixture.load().unwrap();
            let mut installed = entry.clone();
            installed["name"] = spec.name.clone().into();
            installed["geometry"] = spec.geometry.clone();
            installed["capacity_rows"] = spec.capacity_rows.into();
            installed["scratch_bytes_at_capacity"] = serde_json::json!({});
            fixture.register_programs(&library, &[installed.clone()]).unwrap();
            let path = fixture.directory().join("PROGRAMS.json");
            std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"programs":[installed]})).unwrap()).unwrap();
            let programs = library.programs().unwrap().with_manifest(&path).unwrap();
            assert_eq!(ForwardProducer::bind(&programs, kind).unwrap().layout, ForwardLayout { kind, rows: 4096 });
            for field in ["name", "dtype", "shape", "role"] {
                let mut wrong = entry.clone(); wrong["pointers"][0][field] = "wrong".into();
                assert!(validate_entry(&spec, &wrong).is_err());
            }
            let mut wrong = entry; wrong["scalars"][0]["type"] = "int64".into();
            assert!(validate_entry(&spec, &wrong).is_err());
        }
    }
    #[test]
    fn narrow_kv_pack_validates_rank_local_scaled_records_and_extents() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/minimax_forward_programs.json")).unwrap();
        let mut entry = fixture["programs"].as_array().unwrap().iter()
            .find(|e| e["name"] == "minimax_kv_pack").unwrap().clone();
        entry["name"] = "minimax_kv_pack_h2".into();
        entry["geometry"]["heads"] = 2.into();
        entry["geometry"]["record_bytes"] = 272.into();
        entry["geometry"]["scale_offset"] = 16384.into(); entry["geometry"]["scale_stride"] = 16.into();
        for p in entry["pointers"].as_array_mut().unwrap() {
            p["shape"] = p["shape"].as_str().unwrap().replace(",4", ",2").replace(",528]", ",272]").into();
        }
        let spec = spec(Kind::KvPackHeads);
        let layout = validate_entry(&spec, &entry).unwrap();
        let mut bad = spec.clone(); bad.name = Kind::KvPack.name().into();
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone(); bad.geometry["heads"] = 4.into();
        assert!(validate_entry(&bad, &entry).is_err());
        for rows in [1, 63, 64, 2049, 4096] {
            let b = KvBatch { rows, pool_pages: 11,
                key: Buffer { bytes: rows * 512, ..buffer(0) }, value: Buffer { bytes: rows * 512, ..buffer(1) },
                slots: Buffer { bytes: rows * 8, ..buffer(2) },
                key_cache: Buffer { bytes: 11 * 17408, ..buffer(6) },
                value_cache: Buffer { bytes: 11 * 17408, ..buffer(7) } };
            assert!(layout.kv(b).is_ok());
            assert!(ForwardLayout::from_spec(&spec_for_whole_kv()).unwrap().kv(b).is_err());
            for i in 0..5 {
                let mut short = b;
                let target = match i { 0 => &mut short.key, 1 => &mut short.value,
                    2 => &mut short.slots, 3 => &mut short.key_cache, _ => &mut short.value_cache };
                target.bytes -= 1;
                assert!(layout.kv(short).is_err(), "operand {i}");
            }
            assert!(layout.kv(KvBatch { value_cache: b.key_cache, ..b }).is_err());
        }
    }
    fn spec_for_whole_kv() -> ProgramSpec { spec(Kind::KvPack) }
    #[test]
    fn checks_native_operand_extents_devices_and_writable_aliases() {
        let norm = ForwardLayout::from_spec(&spec(Kind::ResidualNorm)).unwrap();
        let mut n = NormBatch { rows: 3, input: buffer(0), weight: buffer(1), output: buffer(2) };
        assert!(norm.norm(n).is_ok());
        n.output.bytes = 3 * 6144 * 2 - 1; assert!(norm.norm(n).is_err());
        n.output = n.input; assert!(norm.norm(n).is_err());
        let router = ForwardLayout::from_spec(&spec(Kind::RouterScores)).unwrap();
        let mut r = RouterBatch { rows: 1, input: buffer(0), weight: buffer(1), output: buffer(2) };
        assert!(router.router(r).is_ok());
        r.weight.bytes = 128 * 6144 * 2; assert!(router.router(r).is_err());
        let kv = ForwardLayout::from_spec(&spec(Kind::KvPack)).unwrap();
        let mut k = KvBatch { rows: 7, pool_pages: 11, key: buffer(0), value: buffer(1),
            slots: buffer(2),
            key_cache: buffer(6), value_cache: buffer(7) };
        assert!(kv.kv(k).is_ok());
        k.slots.device_id = 1; assert!(kv.kv(k).is_err());
        k.slots.device_id = 0; k.value_cache = k.key_cache; assert!(kv.kv(k).is_err());
        k.value_cache = buffer(7); k.pool_pages = usize::MAX; assert!(kv.kv(k).is_err());
        k.pool_pages = 11; k.rows = 4097; assert!(kv.kv(k).is_err());
    }
}
