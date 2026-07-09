use std::ffi::c_void;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use libloading::Library;

// ── Types ────────────────────────────────────────────────────────────────────

pub type GenieStatus = i32;

pub const GENIE_STATUS_SUCCESS: GenieStatus = 0;
pub const GENIE_STATUS_WARNING_ABORTED: GenieStatus = 1;
pub const GENIE_STATUS_WARNING_PAUSED: GenieStatus = 3;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenieDialogConfigHandle(pub *mut c_void);
unsafe impl Send for GenieDialogConfigHandle {}
unsafe impl Sync for GenieDialogConfigHandle {}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenieDialogHandle(pub *mut c_void);
unsafe impl Send for GenieDialogHandle {}
unsafe impl Sync for GenieDialogHandle {}

pub type GenieDialogQueryCallback = unsafe extern "C" fn(
    response: *const c_char,
    sentence_code: i32,
    user_data: *const c_void,
);

// ── API Struct ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GenieApi {
    _lib: Arc<Library>,

    pub get_api_major_version: unsafe extern "C" fn() -> u32,
    pub get_api_minor_version: unsafe extern "C" fn() -> u32,
    pub get_api_patch_version: unsafe extern "C" fn() -> u32,

    pub dialog_config_create_from_json: unsafe extern "C" fn(
        config_json: *const c_char,
        config_handle: *mut GenieDialogConfigHandle,
    ) -> GenieStatus,

    pub dialog_config_free: unsafe extern "C" fn(
        config_handle: GenieDialogConfigHandle,
    ) -> GenieStatus,

    pub dialog_create: unsafe extern "C" fn(
        config_handle: GenieDialogConfigHandle,
        dialog_handle: *mut GenieDialogHandle,
    ) -> GenieStatus,

    pub dialog_query: unsafe extern "C" fn(
        dialog_handle: GenieDialogHandle,
        query_str: *const c_char,
        sentence_code: i32,
        callback: Option<GenieDialogQueryCallback>,
        user_data: *const c_void,
    ) -> GenieStatus,

    pub dialog_reset: unsafe extern "C" fn(
        dialog_handle: GenieDialogHandle,
    ) -> GenieStatus,

    pub dialog_free: unsafe extern "C" fn(
        dialog_handle: GenieDialogHandle,
    ) -> GenieStatus,
}

impl GenieApi {
    pub unsafe fn load(dll_path: &Path) -> Result<Self> {
        let lib = unsafe { Library::new(dll_path)? };
        let lib = Arc::new(lib);

        let (
            get_api_major_version,
            get_api_minor_version,
            get_api_patch_version,
            dialog_config_create_from_json,
            dialog_config_free,
            dialog_create,
            dialog_query,
            dialog_reset,
            dialog_free,
        ) = unsafe {
            (
                *lib.get(b"Genie_getApiMajorVersion")?,
                *lib.get(b"Genie_getApiMinorVersion")?,
                *lib.get(b"Genie_getApiPatchVersion")?,
                *lib.get(b"GenieDialogConfig_createFromJson")?,
                *lib.get(b"GenieDialogConfig_free")?,
                *lib.get(b"GenieDialog_create")?,
                *lib.get(b"GenieDialog_query")?,
                *lib.get(b"GenieDialog_reset")?,
                *lib.get(b"GenieDialog_free")?,
            )
        };

        Ok(Self {
            _lib: lib,
            get_api_major_version,
            get_api_minor_version,
            get_api_patch_version,
            dialog_config_create_from_json,
            dialog_config_free,
            dialog_create,
            dialog_query,
            dialog_reset,
            dialog_free,
        })
    }
}
