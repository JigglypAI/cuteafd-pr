//! Official Step3VL ImagePatcher geometry selected by the Step-5 processor.
//! Pixel filtering remains separate: PIL bilinear cap/crop followed by owner
//! FP32 CLIP normalization and tensor bilinear antialias resize to728/504.
use super::{invalid,Result};

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct StepImagePlan {
    pub source:[u32;2],
    /// Degenerate very thin images are padded at the top-left before cap.
    pub padded:[u32;2],
    pub capped:[u32;2],
    pub crop_canvas:[u32;2],
    pub windows:Vec<[u32;4]>,
    pub newline_after:Vec<bool>,
}
impl StepImagePlan {
    pub fn new(width:u32,height:u32) -> Result<Self> {
        if width==0 || height==0 || u64::from(width)*u64::from(height)>64<<20 {
            return Err(invalid("Step source image must be positive and at most64megapixels"));
        }
        let source=[width,height];
        let ratio=f64::from(width)/f64::from(height);
        let padded=if width.min(height)<32 && !(0.25..=4.0).contains(&ratio) {
            [width.max(height);2]
        } else {source};
        let capped=if padded[0].max(padded[1])>3024 {
            let scale=3024.0/f64::from(padded[0].max(padded[1]));
            [(f64::from(padded[0])*scale) as u32,(f64::from(padded[1])*scale) as u32]
        } else {padded};
        if capped.contains(&0) {return Err(invalid("Step cap produced empty image"));}
        let long=capped[0].max(capped[1]);let short=capped[0].min(capped[1]);
        let ratio=f64::from(long)/f64::from(short);
        let window=if long<728 {if ratio>1.5 {short} else {0}}
            else if ratio>4.0 {short.min(504)} else {504};
        let mut plan=Self {source,padded,capped,crop_canvas:capped,windows:Vec::new(),newline_after:Vec::new()};
        if window==0 {return Ok(plan)}
        let snap=|side:u32| {
            let ratio=f64::from(side)/f64::from(window);
            if ratio<1.0 {side} else {
                let quotient=side/window;
                window*(quotient+u32::from(ratio-f64::from(quotient)>0.2))
            }
        };
        plan.crop_canvas=[snap(capped[0]),snap(capped[1])];
        let starts=|side:u32| {
            let count=if side<=window {1} else {(side-window).div_ceil(window)+1};
            (0..count).map(|i| if i+1==count && count>1 && window*i+window>side {side-window} else {window*i}).collect::<Vec<_>>()
        };
        let nx=if plan.crop_canvas[0]<=window {1} else {(plan.crop_canvas[0]-window).div_ceil(window)+1};
        let ny=if plan.crop_canvas[1]<=window {1} else {(plan.crop_canvas[1]-window).div_ceil(window)+1};
        if u64::from(nx)*u64::from(ny)>128 {return Err(invalid("Step crop count exceeds admitted per-source limit"));}
        let xs=starts(plan.crop_canvas[0]);let ys=starts(plan.crop_canvas[1]);
        for y in ys.iter().copied() {
            for x in xs.iter().copied() {plan.windows.push([x,y,window,window]);}
        }
        plan.newline_after=(0..plan.windows.len()).map(|i| (i+1)%xs.len()==0 && i+1<plan.windows.len()).collect();
        Ok(plan)
    }
    pub fn feature_rows(&self) -> Option<usize> {self.windows.len().checked_mul(81)?.checked_add(169)}

    /// Sources for the owner-side tensor pipeline: preserve the official
    /// intermediate PIL uint8 bilinear cap/crop, never resize to728/504 here.
    pub fn sources(&self,image:super::RgbImage) -> Result<Vec<super::RgbImage>> {
        if [image.width,image.height]!=self.source || image.data.len()!=image.width as usize*image.height as usize*3 {
            return Err(invalid("Step crop source differs from planned RGB dimensions"));
        }
        if u64::from(self.padded[0])*u64::from(self.padded[1])>64<<20 {
            return Err(invalid("Step padded source exceeds admitted64megapixel canvas"));
        }
        let mut padded=vec![0;self.padded[0] as usize*self.padded[1] as usize*3];
        for row in 0..image.height as usize {
            let start=row*self.padded[0] as usize*3;
            padded[start..start+image.width as usize*3].copy_from_slice(&image.data[row*image.width as usize*3..(row+1)*image.width as usize*3]);
        }
        let padded=super::RgbImage {width:self.padded[0],height:self.padded[1],data:padded};
        let global=super::resize::resize(&padded,self.capped[0],self.capped[1],super::Resample::Bilinear);
        let canvas=super::resize::resize(&global,self.crop_canvas[0],self.crop_canvas[1],super::Resample::Bilinear);
        let mut views=Vec::with_capacity(self.windows.len()+1);
        for [x,y,width,height] in self.windows.iter().copied() {
            let mut data=vec![0;width as usize*height as usize*3];
            for row in 0..height as usize {
                let source=((y as usize+row)*canvas.width as usize+x as usize)*3;
                data[row*width as usize*3..(row+1)*width as usize*3].copy_from_slice(&canvas.data[source..source+width as usize*3]);
            }
            views.push(super::RgbImage {width,height,data});
        }
        views.push(global);
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_crop_fixture_matches_executed_vendor_processor() {
        let plan=StepImagePlan::new(896,812).unwrap();
        assert_eq!(plan.crop_canvas,[1008,1008]);
        assert_eq!(plan.windows,vec![[0,0,504,504],[504,0,504,504],[0,504,504,504],[504,504,504,504]]);
        assert_eq!(plan.newline_after,[false,true,false,false]);
        assert_eq!(plan.feature_rows(),Some(493));
    }
    #[test]
    fn crop_geometry_preserves_thresholds_cap_and_raster_order() {
        assert!(StepImagePlan::new(640,480).unwrap().windows.is_empty());
        assert!(!StepImagePlan::new(700,400).unwrap().windows.is_empty());
        assert!(StepImagePlan::new(727,727).unwrap().windows.is_empty());
        assert_eq!(StepImagePlan::new(728,728).unwrap().windows.len(),4);
        let thin=StepImagePlan::new(200,10).unwrap();assert_eq!(thin.padded,[200,200]);assert!(thin.windows.is_empty());
        let wide=StepImagePlan::new(6000,1000).unwrap();assert_eq!(wide.capped,[3024,504]);
        assert_eq!(wide.windows.len(),6);assert!(wide.newline_after.iter().all(|value| !value));
        assert!(StepImagePlan::new(0,1).is_err());
        assert!(StepImagePlan::new(u32::MAX,2).is_err());
    }
}
