use crate::command::args::PixelFormat;
use anyhow::Context;
use clap::Parser;
use std::{borrow::Cow, fmt::Display, sync::Arc, thread};

const DEFAULT_VMAF_FPS: f32 = 25.0;

/// Common vmaf options.
#[derive(Debug, Parser, Clone)]
pub struct Vmaf {
    /// Additional vmaf arg(s). E.g. --vmaf n_threads=8 --vmaf n_subsample=4
    ///
    /// By default `n_threads` is set to available system threads.
    ///
    /// Also see https://ffmpeg.org/ffmpeg-filters.html#libvmaf.
    #[arg(long = "vmaf", value_parser = parse_vmaf_arg)]
    pub vmaf_args: Vec<Arc<str>>,

    /// Video resolution scale to use in VMAF analysis. If set, video streams will be bicubic
    /// scaled to this during VMAF analysis. `auto` (default) automatically sets
    /// based on the model and input video resolution. `none` disables any scaling.
    /// `WxH` format may be used to specify custom scaling, e.g. `1920x1080`.
    ///
    /// auto behaviour:
    /// * 1k model (default for resolutions <= 2560x1440) if width and height
    ///   are less than 1728 & 972 respectively upscale to 1080p. Otherwise no scaling.
    /// * 4k model (default for resolutions > 2560x1440) if width and height
    ///   are less than 3456 & 1944 respectively upscale to 4k. Otherwise no scaling.
    ///
    /// The auto behaviour is based on the distorted video dimensions, equivalent
    /// to post input/reference vfilter dimensions.
    ///
    /// Scaling happens after any input/reference vfilters.
    #[arg(long, default_value_t, value_parser = parse_vmaf_scale)]
    pub vmaf_scale: VmafScale,

    /// Frame rate override used to analyse both reference & distorted videos.
    /// Maps to ffmpeg `-r` input arg.
    ///
    /// Setting to 0 disables use.
    #[arg(long, default_value_t = DEFAULT_VMAF_FPS)]
    pub vmaf_fps: f32,

    /// Use CUDA-accelerated VMAF (libvmaf_cuda) instead of standard libvmaf.
    /// Requires CUDA-capable GPU and ffmpeg built with CUDA support.
    /// When enabled, videos will be decoded and processed on GPU.
    #[arg(long)]
    pub vmaf_cuda: bool,

    /// Precompute the reference filtered sample for VMAF calculation.
    /// The reference sample will be encoded with FFV1 lossless codec with filters applied,
    /// then used for VMAF analysis. This can improve VMAF performance when using complex
    /// filters, as the filtered reference is computed once and reused.
    /// The precomputed sample is deleted after VMAF computation.
    #[arg(long)]
    pub precomp_sample: bool,

    /// Additional ffmpeg input encoder arg(s) for precomputed sample encoding.
    /// E.g. `--precomp-sample-einput hwaccel=vaapi --precomp-sample-einput hwaccel_output_format=vaapi`.
    /// These are added as ffmpeg input file options for the precomputed sample encoding.
    ///
    /// This flag behaves similarly to --enc-input and is only used when --precomp-sample is enabled.
    ///
    /// See --enc-input docs for more details on format.
    #[arg(long = "precomp-sample-einput", allow_hyphen_values = true, value_parser = parse_precomp_einput_arg)]
    pub precomp_sample_einput_args: Vec<String>,
}

fn parse_precomp_einput_arg(arg: &str) -> anyhow::Result<String> {
    let mut arg = arg.to_owned();
    if !arg.starts_with('-') {
        arg.insert(0, '-');
    }
    Ok(arg)
}

impl Default for Vmaf {
    fn default() -> Self {
        Self {
            vmaf_args: <_>::default(),
            vmaf_scale: <_>::default(),
            vmaf_fps: DEFAULT_VMAF_FPS,
            vmaf_cuda: false,
            precomp_sample: false,
            precomp_sample_einput_args: <_>::default(),
        }
    }
}

impl std::hash::Hash for Vmaf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vmaf_args.hash(state);
        self.vmaf_scale.hash(state);
        self.vmaf_fps.to_ne_bytes().hash(state);
        self.vmaf_cuda.hash(state);
        self.precomp_sample.hash(state);
        self.precomp_sample_einput_args.hash(state);
    }
}

fn parse_vmaf_arg(arg: &str) -> anyhow::Result<Arc<str>> {
    Ok(arg.to_owned().into())
}

impl Vmaf {
    pub fn fps(&self) -> Option<f32> {
        Some(self.vmaf_fps).filter(|r| *r > 0.0)
    }

    /// Returns ffmpeg `filter_complex`/`lavfi` value for calculating vmaf.
    /// If `use_precomputed_ref` is true, the reference stream is assumed to already have
    /// filters applied (via precomputed FFV1 sample) and won't have vfilter/scaling applied.
    pub fn ffmpeg_lavfi(
        &self,
        distorted_res: Option<(u32, u32)>,
        pix_fmt: Option<PixelFormat>,
        ref_vfilter: Option<&str>,
        use_precomputed_ref: bool,
    ) -> String {
        if self.vmaf_cuda {
            return self.ffmpeg_lavfi_cuda(distorted_res, ref_vfilter, use_precomputed_ref);
        }

        let mut args = self.vmaf_args.clone();
        if !args.iter().any(|a| a.contains("n_threads")) {
            // default n_threads to all cores
            args.push(
                format!(
                    "n_threads={}",
                    thread::available_parallelism().map_or(1, |p| p.get())
                )
                .into(),
            );
        }
        let mut lavfi = args.join(":");
        lavfi.insert_str(0, "libvmaf=shortest=true:ts_sync_mode=nearest:");

        let mut model = VmafModel::from_args(&args);
        if let (None, Some((w, h))) = (model, distorted_res)
            && w > 2560
            && h > 1440
        {
            // for >2k resolutions use 4k model
            lavfi.push_str(":model=version=vmaf_4k_v0.6.1");
            model = Some(VmafModel::Vmaf4K);
        }

        let format = pix_fmt.map(|v| format!("format={v},")).unwrap_or_default();
        let scale = self
            .vf_scale(model.unwrap_or_default(), distorted_res)
            .map(|(w, h)| format!("scale={w}:{h}:flags=bicubic,"))
            .unwrap_or_default();

        let prefix = if use_precomputed_ref {
            // When using precomputed ref, filters already applied to reference
            // Only process distorted stream
            format!(
                "[0:v]{format}{scale}setpts=PTS-STARTPTS,settb=AVTB[dis];\
                 [1:v]setpts=PTS-STARTPTS,settb=AVTB[ref];\
                 [dis][ref]"
            )
        } else {
            // Normal mode: apply vfilter and scaling to both streams
            let ref_vf: Cow<_> = match ref_vfilter {
                None => "".into(),
                Some(vf) if vf.ends_with(',') => vf.into(),
                Some(vf) => format!("{vf},").into(),
            };
            
            format!(
                "[0:v]{format}{scale}setpts=PTS-STARTPTS,settb=AVTB[dis];\
                 [1:v]{format}{ref_vf}{scale}setpts=PTS-STARTPTS,settb=AVTB[ref];\
                 [dis][ref]"
            )
        };

        lavfi.insert_str(0, &prefix);
        lavfi
    }

    /// Returns ffmpeg `filter_complex` value for calculating vmaf with CUDA acceleration.
    /// Heavily duplicated from the above, will refactor later.
    fn ffmpeg_lavfi_cuda(
        &self,
        distorted_res: Option<(u32, u32)>,
        ref_vfilter: Option<&str>,
        use_precomputed_ref: bool,
    ) -> String {
        let args = self.vmaf_args.clone();
        let mut lavfi = if args.is_empty() {
            "libvmaf_cuda=ts_sync_mode=nearest:shortest=true".to_string()
        } else {
            format!("libvmaf_cuda={}", args.join(":"))
        };

        let mut model = VmafModel::from_args(&args);
        if let (None, Some((w, h))) = (model, distorted_res)
            && w > 2560
            && h > 1440
        {
            // for >2k resolutions use 4k model
            lavfi.push_str(":model=version=vmaf_4k_v0.6.1");
            model = Some(VmafModel::Vmaf4K);
        }

        // For CUDA, we use scale_cuda instead of scale
        let scale = self
            .vf_scale(model.unwrap_or_default(), distorted_res)
            .map(|(w, h)| {
                // scale_cuda needs explicit format specification
                format!("scale_npp={}:{}:format=yuv420p:interp_algo=lanczos", w, h)
            })
            .unwrap_or_else(|| "scale_npp=format=yuv420p:interp_algo=lanczos".to_string());
            // 

        let prefix = if use_precomputed_ref {
            // When using precomputed ref with CUDA:
            // - Precomputed FFV1 is decoded on CPU
            // - Need to upload to GPU with hwupload_cuda
            // - Then convert format with scale_npp
            let ref_upload = "hwupload_cuda,scale_npp=format=yuv420p:interp_algo=lanczos";
            format!(
                "[0:v]{scale}[dis];\
                 [1:v]{ref_upload}[ref];\
                 [dis][ref]"
            )
        } else {
            // CUDA pipeline:
            // * Decode on GPU (handled by input args in vmaf.rs)
            // * Apply reference vfilter if any
            // * Scale to yuv420p on GPU
            // * Process with libvmaf_cuda
            let ref_vf: Cow<_> = match ref_vfilter {
                None => "".into(),
                Some(vf) if vf.ends_with(',') => vf.into(),
                Some(vf) => format!("{vf},").into(),
            };
            
            format!(
                "[0:v]{scale}[dis];\
                 [1:v]{ref_vf}{scale}[ref];\
                 [dis][ref]"
            )
        };

        lavfi.insert_str(0, &prefix);
        lavfi
    }

    fn vf_scale(&self, model: VmafModel, distorted_res: Option<(u32, u32)>) -> Option<(i32, i32)> {
        match (self.vmaf_scale, distorted_res) {
            (VmafScale::Auto, Some((w, h))) => match model {
                // upscale small resolutions to 1k for use with the 1k model
                VmafModel::Vmaf1K if w < 1728 && h < 972 => {
                    Some(minimally_scale((w, h), (1920, 1080)))
                }
                // upscale small resolutions to 4k for use with the 4k model
                VmafModel::Vmaf4K if w < 3456 && h < 1944 => {
                    Some(minimally_scale((w, h), (3840, 2160)))
                }
                _ => None,
            },
            (VmafScale::Custom { width, height }, Some((w, h))) => {
                Some(minimally_scale((w, h), (width, height)))
            }
            (VmafScale::Custom { width, height }, None) => Some((width as _, height as _)),
            _ => None,
        }
    }
}

/// Return the smallest ffmpeg vf `(w, h)` scale values so that at least one of the
/// `target_w` or `target_h` bounds are met.
fn minimally_scale((from_w, from_h): (u32, u32), (target_w, target_h): (u32, u32)) -> (i32, i32) {
    let w_factor = from_w as f64 / target_w as f64;
    let h_factor = from_h as f64 / target_h as f64;
    if h_factor > w_factor {
        (-1, target_h as _) // scale vertically
    } else {
        (target_w as _, -1) // scale horizontally
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VmafScale {
    None,
    #[default]
    Auto,
    Custom {
        width: u32,
        height: u32,
    },
}

fn parse_vmaf_scale(vs: &str) -> anyhow::Result<VmafScale> {
    const ERR: &str = "vmaf-scale must be 'none', 'auto' or WxH format e.g. '1920x1080'";
    match vs {
        "none" => Ok(VmafScale::None),
        "auto" => Ok(VmafScale::Auto),
        _ => {
            let (w, h) = vs.split_once('x').context(ERR)?;
            let (width, height) = (w.parse().context(ERR)?, h.parse().context(ERR)?);
            Ok(VmafScale::Custom { width, height })
        }
    }
}

impl Display for VmafScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => "none".fmt(f),
            Self::Auto => "auto".fmt(f),
            Self::Custom { width, height } => write!(f, "{width}x{height}"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
enum VmafModel {
    /// Default 1080p model.
    #[default]
    Vmaf1K,
    /// 4k model.
    Vmaf4K,
    /// Some other user specified model.
    Custom,
}

impl VmafModel {
    fn from_args(args: &[Arc<str>]) -> Option<Self> {
        let mut using_custom_model: Vec<_> = args.iter().filter(|v| v.contains("model")).collect();

        match using_custom_model.len() {
            0 => None,
            1 => Some(match using_custom_model.remove(0) {
                v if v.ends_with("version=vmaf_v0.6.1") => Self::Vmaf1K,
                v if v.ends_with("version=vmaf_4k_v0.6.1") => Self::Vmaf4K,
                _ => Self::Custom,
            }),
            _ => Some(Self::Custom),
        }
    }
}

#[test]
fn vmaf_lavfi() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(
            None,
            Some(PixelFormat::Yuv420p),
            Some("scale=1280:-1,fps=24"),
            false
        ),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=1280:-1,fps=24,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_default() {
    let vmaf = Vmaf::default();
    let expected = format!(
        "[0:v]setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(vmaf.ffmpeg_lavfi(None, None, None, false), expected);
}

#[test]
fn vmaf_lavfi_default_pix_fmt() {
    let vmaf = Vmaf::default();
    let expected = format!(
        "[0:v]format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(
        vmaf.ffmpeg_lavfi(None, Some(PixelFormat::Yuv420p10le), None, false),
        expected
    );
}

#[test]
fn vmaf_lavfi_include_n_threads() {
    let vmaf = Vmaf {
        vmaf_args: vec!["log_path=output.xml".into()],
        ..<_>::default()
    };
    let expected = format!(
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:log_path=output.xml:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(
        vmaf.ffmpeg_lavfi(None, Some(PixelFormat::Yuv420p), None, false),
        expected
    );
}

/// Low resolution videos should be upscaled to 1080p
#[test]
fn vmaf_lavfi_small_width() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}

/// 4k videos should use 4k model
#[test]
fn vmaf_lavfi_4k() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((3840, 2160)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4:model=version=vmaf_4k_v0.6.1"
    );
}

/// >2k videos should be upscaled to 4k & use 4k model
#[test]
fn vmaf_lavfi_3k_upscale_to_4k() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((3008, 1692)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:model=version=vmaf_4k_v0.6.1"
    );
}

/// If user has overridden the model, don't default a vmaf width
#[test]
fn vmaf_lavfi_small_width_custom_model() {
    let vmaf = Vmaf {
        vmaf_args: vec![
            "model=version=foo".into(),
            "n_threads=5".into(),
            "n_subsample=4".into(),
        ],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:model=version=foo:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_custom_model_and_width() {
    let vmaf = Vmaf {
        vmaf_args: vec![
            "model=version=foo".into(),
            "n_threads=5".into(),
            "n_subsample=4".into(),
        ],
        // if specified just do it
        vmaf_scale: VmafScale::Custom {
            width: 123,
            height: 720,
        },
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,scale=123:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=123:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:model=version=foo:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_1080p() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1920, 1080)), Some(PixelFormat::Yuv420p), None, false),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_cuda_1080p() {
    let vmaf = Vmaf {
        vmaf_args: vec!["log_fmt=json:log_path=output.json".into()],
        vmaf_cuda: true,
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1920, 1080)), None, None, false),
        "[0:v]scale_npp=format=yuv420p:interp_algo=lanczos[dis];\
         [1:v]scale_npp=format=yuv420p:interp_algo=lanczos[ref];\
         [dis][ref]libvmaf_cuda=log_fmt=json:log_path=output.json"
    );
}

#[test]
fn vmaf_lavfi_cuda_720p_upscale() {
    let vmaf = Vmaf {
        vmaf_args: vec![],
        vmaf_cuda: true,
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), None, None, false),
        "[0:v]scale_npp=1920:-1:format=yuv420p:interp_algo=lanczos[dis];\
         [1:v]scale_npp=1920:-1:format=yuv420p:interp_algo=lanczos[ref];\
         [dis][ref]libvmaf_cuda=ts_sync_mode=nearest:shortest=true"
    );
}

#[test]
fn vmaf_lavfi_cuda_4k() {
    let vmaf = Vmaf {
        vmaf_args: vec![],
        vmaf_cuda: true,
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((3840, 2160)), None, None, false),
        "[0:v]scale_npp=format=yuv420p:interp_algo=lanczos[dis];\
         [1:v]scale_npp=format=yuv420p:interp_algo=lanczos[ref];\
         [dis][ref]libvmaf_cuda=ts_sync_mode=nearest:shortest=true:model=version=vmaf_4k_v0.6.1"
    );
}

#[test]
fn vmaf_lavfi_cuda_precomputed_ref() {
    let vmaf = Vmaf {
        vmaf_args: vec![],
        vmaf_cuda: true,
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1920, 1080)), None, None, true),
        "[0:v]scale_npp=format=yuv420p:interp_algo=lanczos[dis];\
         [1:v]scale_npp=format=yuv420p:interp_algo=lanczos[ref];\
         [dis][ref]libvmaf_cuda=ts_sync_mode=nearest:shortest=true"
    );
}

#[test]
fn vmaf_lavfi_precomputed_ref() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1920, 1080)), Some(PixelFormat::Yuv420p), None, true),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5"
    );
}

