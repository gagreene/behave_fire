//! Reuse the existing owned typed buffers rather than introducing parallel containers.

/// Surface columns with checked shape, optional native validity, and batch statistics.
pub type SurfaceRasterOutput = crate::batch::SurfaceBatch;
/// Crown columns with i32 classification and optional native validity.
pub type CrownRasterOutput = crate::batch::CrownBatch;
