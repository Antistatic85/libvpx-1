/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_TOKENIZE_H_
#define VP9_ENCODER_VP9_TOKENIZE_H_

#include "vp9/common/vp9_entropy.h"

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_treewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EOSB_TOKEN 15     // Not signalled, encoder only

#if CONFIG_VP9_HIGHBITDEPTH
  typedef int32_t EXTRABIT;
#else
  typedef int16_t EXTRABIT;
#endif


typedef struct {
  int16_t token;
  EXTRABIT extra;
} TOKENVALUE;

typedef struct {
  uint8_t token : 4;
  // coeff_type ranges from 0 - PLANE_TYPES
  uint8_t coeff_type : 1;
  // coeff_band ranges from 0 - COEF_BANDS
  uint8_t coeff_band : 3;
  // coeff_ctx ranges from 0 - COEFF_CONTEXTS * UNCONSTRAINED_NODES
  uint8_t coeff_ctx  : 5;
  // tx_size ranges from 0 - TX_SIZES
  uint8_t tx_size : 2;
  uint8_t skip_eob_node : 1;

  EXTRABIT extra;
} TOKENEXTRA;

typedef struct {
  TOKENEXTRA *start;
  TOKENEXTRA *stop;
} TOKENLIST;

extern const vpx_tree_index vp9_coef_tree[];
extern const vpx_tree_index vp9_coef_con_tree[];
extern const struct vp9_token vp9_coef_encodings[];

int vp9_is_skippable_in_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane);
int vp9_has_high_freq_in_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane);

struct VP9_COMP;
struct ThreadData;

void vp9_tokenize_sb(struct VP9_COMP *cpi, struct ThreadData *td,
                     TOKENEXTRA **t, int dry_run, BLOCK_SIZE bsize);

typedef struct {
  const vpx_prob *prob;
  int len;
  int base_val;
  const int16_t *cost;
} vp9_extra_bit;

// indexed by token value
extern const vp9_extra_bit vp9_extra_bits[ENTROPY_TOKENS];
#if CONFIG_VP9_HIGHBITDEPTH
extern const vp9_extra_bit vp9_extra_bits_high10[ENTROPY_TOKENS];
extern const vp9_extra_bit vp9_extra_bits_high12[ENTROPY_TOKENS];
#endif  // CONFIG_VP9_HIGHBITDEPTH

extern const int16_t *vp9_dct_value_cost_ptr;
/* TODO: The Token field should be broken out into a separate char array to
 *  improve cache locality, since it's needed for costing when the rest of the
 *  fields are not.
 */
extern const TOKENVALUE *vp9_dct_value_tokens_ptr;
extern const TOKENVALUE *vp9_dct_cat_lt_10_value_tokens;
extern const int16_t vp9_cat6_low_cost[256];
extern const int16_t vp9_cat6_high_cost[128];
extern const int16_t vp9_cat6_high10_high_cost[512];
extern const int16_t vp9_cat6_high12_high_cost[2048];
static INLINE int16_t vp9_get_cost(int16_t token, EXTRABIT extrabits,
                                   const int16_t *cat6_high_table) {
  if (token != CATEGORY6_TOKEN)
    return vp9_extra_bits[token].cost[extrabits];
  return vp9_cat6_low_cost[extrabits & 0xff]
      + cat6_high_table[extrabits >> 8];
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE const int16_t* vp9_get_high_cost_table(int bit_depth) {
  return bit_depth == 8 ? vp9_cat6_high_cost
      : (bit_depth == 10 ? vp9_cat6_high10_high_cost :
         vp9_cat6_high12_high_cost);
}
#else
static INLINE const int16_t* vp9_get_high_cost_table(int bit_depth) {
  (void) bit_depth;
  return vp9_cat6_high_cost;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static INLINE void vp9_get_token_extra(int v, int16_t *token, EXTRABIT *extra) {
  if (v >= CAT6_MIN_VAL || v <= -CAT6_MIN_VAL) {
    *token = CATEGORY6_TOKEN;
    if (v >= CAT6_MIN_VAL)
      *extra = 2 * v - 2 * CAT6_MIN_VAL;
    else
      *extra = -2 * v - 2 * CAT6_MIN_VAL + 1;
    return;
  }
  *token = vp9_dct_cat_lt_10_value_tokens[v].token;
  *extra = vp9_dct_cat_lt_10_value_tokens[v].extra;
}
static INLINE int16_t vp9_get_token(int v) {
  if (v >= CAT6_MIN_VAL || v <= -CAT6_MIN_VAL)
    return 10;
  return vp9_dct_cat_lt_10_value_tokens[v].token;
}


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_TOKENIZE_H_
