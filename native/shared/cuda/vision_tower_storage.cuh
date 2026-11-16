#pragma once
#include <stdint.h>

struct GenericTowerStorage {
  uint16_t *x=nullptr,*next=nullptr,*norm=nullptr,*qkv_norm=nullptr,*patches=nullptr,*qkv=nullptr;
  uint16_t *q=nullptr,*k=nullptr,*v=nullptr,*attention=nullptr,*projection=nullptr;
  uint16_t *mlp=nullptr,*position=nullptr,*grouped=nullptr,*projector=nullptr,*projector_result=nullptr,*postnorm_input=nullptr,*output=nullptr;
  uint16_t *preprojector=nullptr,*preprojected=nullptr;
  float *cosines=nullptr,*sines=nullptr,*lse=nullptr,*lut=nullptr;
  uint8_t* rgb=nullptr;
  int32_t* lengths=nullptr;
};
