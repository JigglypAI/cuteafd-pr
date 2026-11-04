//! dSpark bindings to the shared shifted mHC boundary owner.
use super::DsparkWeights;
pub(crate) use crate::families::deepseek_v41::v41_hc::HcSublayer;
use anyhow::{ensure, Result};
impl<'library> DsparkWeights<'library> {
    pub fn hc_sublayer(
        &self,
        stage: usize,
        attention: bool,
        capacity: usize,
        budget: usize,
    ) -> Result<HcSublayer<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark stage");
        let kind = if attention { "attn" } else { "ffn" };
        let names = [
            format!("mtp.{stage}.hc_{kind}_fn"),
            format!("mtp.{stage}.hc_{kind}_scale"),
            format!("mtp.{stage}.hc_{kind}_base"),
            format!("mtp.{stage}.{kind}_norm.weight"),
        ];
        HcSublayer::new(
            self.library,
            &self.auxiliary,
            names,
            capacity,
            budget,
        )
    }
    pub(super) fn hc_sublayer_on(&self, stage: usize, attention: bool, capacity: usize,
        stream: std::rc::Rc<crate::shared::memory::LoadStream<'library>>) -> Result<HcSublayer<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark stage");
        let kind = if attention { "attn" } else { "ffn" };
        let names = [format!("mtp.{stage}.hc_{kind}_fn"), format!("mtp.{stage}.hc_{kind}_scale"),
            format!("mtp.{stage}.hc_{kind}_base"), format!("mtp.{stage}.{kind}_norm.weight")];
        HcSublayer::on_storage(self.library, &self.auxiliary, names,
            crate::families::deepseek_v41::v41_hc::HcStorage::with_stream(self.library, capacity, stream)?, capacity)
    }

}
