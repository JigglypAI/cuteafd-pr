//! One source image can produce several independent encoder jobs and prompt
//! spans. Keep crop keys separate: the existing shared media admission/cache
//! already deduplicates jobs and attaches their leases without a feature copy.
use super::{invalid,ExpandedMediaPrompt,ImageKey,MediaSpan,PreparedImage,Result};
use std::sync::Arc;

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum ImageFragment {
    /// Native template/control ids, not user text or synthetic media ids.
    Tokens(Vec<u32>),
    /// Feature rows of one independently cached crop/view.
    Features {image:usize,placeholder:u32},
}
#[derive(Clone,Debug)]
pub struct PreparedImageGroup {
    pub source:[u32;2],
    pub images:Vec<Arc<PreparedImage>>,
    pub fragments:Vec<ImageFragment>,
}
impl PreparedImageGroup {
    pub fn single(image:Arc<PreparedImage>,placeholder:u32) -> Self {
        Self {source:image.pixels.source,images:vec![image],
            fragments:vec![ImageFragment::Features {image:0,placeholder}]}
    }
    pub fn feature_rows(&self) -> Result<usize> {
        self.images.iter().try_fold(0usize,|rows,image| rows.checked_add(image.tokens)
            .ok_or_else(|| invalid("image group feature count overflow")))
    }
    pub fn validate(&self,vocabulary:u32) -> Result<usize> {
        if self.source.contains(&0) || self.images.is_empty() || self.images.len()>128
            || self.fragments.is_empty() || self.fragments.len()>1024 || vocabulary==0 || vocabulary>=1<<31 {
            return Err(invalid("invalid image group source/count/vocabulary"));
        }
        let mut seen=vec![false;self.images.len()];
        let mut length=0usize;
        for fragment in &self.fragments {
            let rows=match fragment {
                ImageFragment::Tokens(tokens) => {
                    if tokens.is_empty() || tokens.iter().any(|id| *id>=vocabulary) {
                        return Err(invalid("image group marker outside checkpoint vocabulary"));
                    }
                    tokens.len()
                }
                ImageFragment::Features {image,placeholder} => {
                    let view=self.images.get(*image).ok_or_else(|| invalid("image group crop index outside prepared jobs"))?;
                    if *placeholder>=vocabulary || view.tokens==0 || seen[*image] || !view.pixels.valid()
                        || view.pixels.source!=self.source || view.grid.t!=1 || view.grid.h==0 || view.grid.w==0
                        || u64::from(view.pixels.canvas[0])*u64::from(view.pixels.canvas[1])*3!=view.rgb8.len() as u64 {
                        return Err(invalid("image group crop geometry/order differs from prepared input"));
                    }
                    seen[*image]=true;
                    view.tokens
                }
            };
            length=length.checked_add(rows).ok_or_else(|| invalid("image group prompt length overflow"))?;
        }
        if seen.iter().any(|seen| !seen) {return Err(invalid("image group has unused encoder jobs"));}
        Ok(length)
    }

    /// Replaces one source-image placeholder with arbitrary ordered fragments.
    /// Feature fragments create the same per-key spans existing media_pending
    /// consumes; token-only separators do not become injected feature rows.
    pub fn expand_prompt(tokens:&[u32],source_placeholder:u32,groups:&[Self],vocabulary:u32,context:usize)
        -> Result<ExpandedMediaPrompt> {
        if vocabulary==0 || vocabulary>=1<<31 || source_placeholder>=vocabulary
            || tokens.iter().any(|id| *id>=vocabulary) {
            return Err(invalid("image group prompt token outside checkpoint vocabulary"));
        }
        if tokens.iter().filter(|id| **id==source_placeholder).count()!=groups.len() {
            return Err(invalid("source image placeholder count differs from prepared groups"));
        }
        let mut length=tokens.len();
        for group in groups {
            let rows=group.validate(vocabulary)?;
            length=length.checked_add(rows-1).ok_or_else(|| invalid("grouped image prompt length overflow"))?;
        }
        if length>context {return Err(invalid("grouped image prompt exceeds context limit"));}
        let mut expanded=Vec::with_capacity(length);
        let mut spans=Vec::new();let mut source=0;
        for token in tokens {
            if *token!=source_placeholder {expanded.push(*token);continue;}
            let group=&groups[source];source+=1;
            for fragment in &group.fragments {
                match fragment {
                    ImageFragment::Tokens(ids)=>expanded.extend_from_slice(ids),
                    ImageFragment::Features {image,placeholder}=> {
                        let image=&group.images[*image];
                        spans.push(MediaSpan {start:expanded.len(),len:image.tokens,key:image.key.into()});
                        expanded.extend(std::iter::repeat_n(*placeholder,image.tokens));
                    }
                }
            }
        }
        Ok(ExpandedMediaPrompt {tokens:expanded,media:spans})
    }
    pub fn keys_in_prompt_order(&self) -> impl Iterator<Item=ImageKey>+'_ {
        self.fragments.iter().filter_map(|fragment| match fragment {
            ImageFragment::Features {image,..}=>self.images.get(*image).map(|image|image.key),_=>None})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{EncoderId,ImageGrid,PreprocessId};
    fn view(key:u8,tokens:usize,source:[u32;2]) -> Arc<PreparedImage> {
        Arc::new(PreparedImage {key:ImageKey([key;32]),encoder:EncoderId([0;32]),processor:PreprocessId([0;32]),
            grid:ImageGrid {t:1,h:4,w:4},rgb8:vec![0;56*56*3].into(),tokens,
            pixels:cuteafd_core::media::ImagePixels {source,..cuteafd_core::media::ImagePixels::full(56,56)}})
    }
    fn group() -> PreparedImageGroup {
        PreparedImageGroup {source:[896,812],images:vec![view(1,81,[896,812]),view(2,81,[896,812]),view(3,169,[896,812])],
            fragments:vec![ImageFragment::Tokens(vec![10]),ImageFragment::Features {image:0,placeholder:7},
                ImageFragment::Tokens(vec![11,12,10]),ImageFragment::Features {image:1,placeholder:7},
                ImageFragment::Tokens(vec![11,13]),ImageFragment::Features {image:2,placeholder:7},ImageFragment::Tokens(vec![14])]}
    }
    #[test]
    fn local_before_global_spans_keep_token_only_separators_out_of_features() {
        let group=group();assert_eq!(group.feature_rows().unwrap(),331);
        let expanded=PreparedImageGroup::expand_prompt(&[1,7,2],7,&[group.clone()],32,1024).unwrap();
        assert_eq!(expanded.media.iter().map(|span|(span.start,span.len)).collect::<Vec<_>>(),vec![(2,81),(86,81),(169,169)]);
        assert_eq!(expanded.tokens.len(),340);
        assert_eq!(expanded.tokens[83..86],[11,12,10]);
        assert_eq!(expanded.tokens[167..169],[11,13]);
        assert_eq!(group.keys_in_prompt_order().collect::<Vec<_>>(),vec![ImageKey([1;32]),ImageKey([2;32]),ImageKey([3;32])]);
    }
    #[test]
    fn group_order_repetition_and_context_are_checked_before_encoding() {
        let group=group();
        assert!(PreparedImageGroup::expand_prompt(&[7],7,&[group.clone()],32,337).is_err());
        let mut wrong=group.clone();wrong.fragments.push(ImageFragment::Features {image:0,placeholder:7});
        assert!(wrong.validate(32).is_err());
        let mut wrong=group.clone();wrong.fragments.retain(|f| !matches!(f,ImageFragment::Features {image:2,..}));
        assert!(wrong.validate(32).is_err());
        let mut wrong=group.clone();wrong.fragments.push(ImageFragment::Tokens(vec![32]));assert!(wrong.validate(32).is_err());
        let mut wrong=group;wrong.images[0]=view(1,81,[100,100]);assert!(wrong.validate(32).is_err());
    }
}
