// Included only by the oracle TU; production has no symbols bypassing admission.
extern "C" int32_t cuteafd_vision_oracle_required(const cuteafd_vision_spec* spec,
    cuteafd_vision_ledger* ledger) {
  return vision_required_impl(spec,ledger,valid_generic_tower);
}
extern "C" int32_t cuteafd_vision_oracle_create(const cuteafd_vision_spec* spec,
    int32_t device,uint64_t admitted,const cuteafd_media_tags* tags,void** owner) {
  return vision_create_impl(spec,device,admitted,tags,owner,valid_generic_tower);
}
