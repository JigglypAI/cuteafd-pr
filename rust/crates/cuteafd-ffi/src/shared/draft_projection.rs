//! Preloaded BF16 drafter projections with one canonical per-row K reduction.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use std::{collections::BTreeMap, ffi::c_void};
use thiserror::Error;

pub const MAX_ROWS: usize = 2048;

#[derive(Debug, Error)]
pub enum ProjectionError {
    #[error("BF16 drafter projection K={k}, outputs={n} is missing or invalid: {reason}")]
    Shape { k: usize, n: usize, reason: String },
    #[error("BF16 drafter projection invalid buffers or rows")]
    Arguments,
    #[error("BF16 drafter projection launch: {0}")]
    Launch(#[from] anyhow::Error),
}

type Result<T> = std::result::Result<T, ProjectionError>;

/// Resolves every admitted shape before capture; launch never loads programs.
pub struct DraftProjections<'a> {
    programs: BTreeMap<(usize, usize), Program<'a>>,
}

impl<'a> DraftProjections<'a> {
    pub fn load(programs: &'a Programs<'a>, shapes: &[(usize, usize)]) -> Result<Self> {
        let mut loaded = BTreeMap::new();
        for &(k, n) in shapes {
            if loaded.contains_key(&(k, n)) { continue; }
            let name = format!("draft_bf16_k{k}_n{n}");
            let invalid = |reason: String| ProjectionError::Shape { k, n, reason };
            let spec = programs.spec(&name).map_err(|e| invalid(e.to_string()))?;
            validate(spec, k, n).map_err(invalid)?;
            let program = programs.program(&name, &["x", "w", "out"])
                .map_err(|e| ProjectionError::Shape { k, n, reason: e.to_string() })?;
            loaded.insert((k, n), program);
        }
        Ok(Self { programs: loaded })
    }

    /// # Safety
    /// Device buffers hold nonoverlapping BF16 [rows,k], [n,k] and [rows,n]
    /// on the supplied stream. Their owners survive all queued work.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(&self, x: *const c_void, weight: *const c_void, out: *mut c_void,
        rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()> {
        if !(1..=MAX_ROWS).contains(&rows) || [x, weight, out.cast_const()].iter()
            .any(|p| p.is_null() || *p as usize % 16 != 0) {
            return Err(ProjectionError::Arguments);
        }
        let program = self.programs.get(&(k, n)).ok_or_else(|| ProjectionError::Shape {
            k, n, reason: "shape was not admitted at startup".into() })?;
        // SAFETY: the caller retains the documented buffers and stream; shape and rows are checked.
        unsafe { program.launch(&[x.cast_mut(), weight.cast_mut(), out], &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
}

fn validate(spec: &ProgramSpec, k: usize, n: usize) -> std::result::Result<(), String> {
    let g = &spec.geometry;
    if k == 0 || n == 0 || k % 8 != 0 || n % 8 != 0
        || spec.name != format!("draft_bf16_k{k}_n{n}") || spec.capacity_rows as usize != MAX_ROWS
        || spec.scalar_kinds != b"i" || !spec.scratch.is_empty()
        || !spec.pointers.iter().map(String::as_str).eq(["x", "w", "out"])
        || g["width"].as_u64() != Some(k as u64) || g["outputs"].as_u64() != Some(n as u64)
        || g["output_dtype"].as_str() != Some("torch.bfloat16")
        || g["reduction"].as_str() != Some("ordered_k16")
        || g["tiles_by_rows"] != serde_json::json!([[16, 1, 4, 16, 64, 6]]) {
        return Err("expected canonical K16 BF16 ABI, M16N64 small-row route and capacity 2048".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec { name: "draft_bf16_k5120_n7168".into(), pointers: vec!["x".into(), "w".into(), "out".into()],
            scalar_kinds: b"i".to_vec(), scratch: Default::default(), capacity_rows: MAX_ROWS as u32,
            geometry: serde_json::json!({"width":5120,"outputs":7168,"output_dtype":"torch.bfloat16",
                "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}) }
    }
    #[test]
    fn canonical_manifest_is_required() {
        assert!(validate(&spec(),5120,7168).is_ok());
        for field in ["width","outputs","output_dtype","reduction","tiles_by_rows"] {
            let mut s=spec(); s.geometry[field]=serde_json::Value::Null;
            assert!(validate(&s,5120,7168).is_err(), "{field}");
        }
        let mut s=spec(); s.capacity_rows=140; assert!(validate(&s,5120,7168).is_err());
        let mut s=spec(); s.pointers.swap(0,1); assert!(validate(&s,5120,7168).is_err());
    }
}
