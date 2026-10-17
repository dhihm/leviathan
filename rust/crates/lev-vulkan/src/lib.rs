//! Vulkan compute backend: device selection, a bounded Q4_K matvec runner with
//! GPU timestamps, and content-addressed SPIR-V caching.

use std::ffi::{c_void, CStr};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ash::{vk, Device, Entry, Instance};
use lev_forge::glsl::{group_count, source_digest};
use lev_forge::schedule::{DeviceCaps, Q4kSchedule};

mod engine;
pub use engine::{Buf, Kernel, Program, Step};

const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
/// Failure records keep at most this much compiler output.
const MAX_LOG_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub struct VkError(pub String);

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VkError {}

impl From<vk::Result> for VkError {
    fn from(r: vk::Result) -> Self {
        VkError(format!("vulkan call failed: {r:?}"))
    }
}

impl From<std::io::Error> for VkError {
    fn from(e: std::io::Error) -> Self {
        VkError(format!("io error: {e}"))
    }
}

fn err<T>(m: impl Into<String>) -> Result<T, VkError> {
    Err(VkError(m.into()))
}

#[derive(Clone, Debug)]
pub enum DeviceSelector {
    /// Prefer a discrete GPU, then an integrated one, then the first device.
    Auto,
    Index(usize),
    NameContains(String),
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub device_type: String,
    pub caps: DeviceCaps,
    pub max_group_count_x: u32,
    pub timestamp_period_ns: f32,
    pub timestamp_valid_bits: u32,
    /// VK_KHR_shader_integer_dot_product is available with the feature set.
    pub int_dot: bool,
    /// The packed 4x8-bit signed dot product is hardware accelerated.
    pub int_dot_accelerated: bool,
    /// VK_KHR_cooperative_matrix with the features the matrix kernels need is enabled.
    pub coopmat: bool,
    /// Names of device extensions of interest that the driver exposes.
    pub extensions: Vec<String>,
}

pub struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    phys: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    info: DeviceInfo,
}

fn device_type_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn describe(instance: &Instance, phys: vk::PhysicalDevice, index: usize, queue_family: Option<u32>) -> DeviceInfo {
    let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub);
    unsafe { instance.get_physical_device_properties2(phys, &mut props2) };
    let props = props2.properties;
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned();
    let valid_bits = queue_family
        .map(|qf| {
            let q = unsafe { instance.get_physical_device_queue_family_properties(phys) };
            q[qf as usize].timestamp_valid_bits
        })
        .unwrap_or(0);
    let mut int_dot = false;
    let mut int_dot_accelerated = false;
    let exts = unsafe { instance.enumerate_device_extension_properties(phys) }.unwrap_or_default();
    if exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == c"VK_KHR_shader_integer_dot_product") {
        let mut f = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
        unsafe { instance.get_physical_device_features2(phys, &mut f2) };
        int_dot = f.shader_integer_dot_product == vk::TRUE;
        let mut p = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p);
        unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
        int_dot_accelerated = p.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE;
    }
    let extensions: Vec<String> = exts
        .iter()
        .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }.to_string_lossy().into_owned())
        .filter(|n| n.contains("subgroup_size") || n.contains("cooperative") || n.contains("integer_dot") || n.contains("shader_float16") || n.contains("16bit") || n.contains("8bit") || n.contains("subgroup") || n.contains("maintenance") || n.contains("pipeline_executable"))
        .collect();
    DeviceInfo {
        index,
        name,
        device_type: device_type_name(props.device_type).to_string(),
        caps: DeviceCaps {
            subgroup_size: sub.subgroup_size,
            max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
        },
        max_group_count_x: props.limits.max_compute_work_group_count[0],
        timestamp_period_ns: props.limits.timestamp_period,
        timestamp_valid_bits: valid_bits,
        int_dot,
        int_dot_accelerated,
        coopmat: false,
        extensions,
    }
}

fn compute_queue_family(instance: &Instance, phys: vk::PhysicalDevice) -> Option<u32> {
    let q = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    q.iter()
        .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE) && p.timestamp_valid_bits > 0)
        .map(|i| i as u32)
}

impl VulkanContext {
    pub fn list_devices() -> Result<Vec<DeviceInfo>, VkError> {
        let (entry, instance) = create_instance()?;
        let phys = unsafe { instance.enumerate_physical_devices() }?;
        let out = phys
            .iter()
            .enumerate()
            .map(|(i, &p)| describe(&instance, p, i, compute_queue_family(&instance, p)))
            .collect();
        unsafe { instance.destroy_instance(None) };
        drop(entry);
        Ok(out)
    }

    pub fn open(selector: &DeviceSelector) -> Result<Self, VkError> {
        let (entry, instance) = create_instance()?;
        match Self::open_with(entry, &instance, selector) {
            Ok(ctx) => Ok(ctx),
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn open_with(entry: Entry, instance: &Instance, selector: &DeviceSelector) -> Result<Self, VkError> {
        let phys_all = unsafe { instance.enumerate_physical_devices() }?;
        let mut infos: Vec<DeviceInfo> = phys_all
            .iter()
            .enumerate()
            .map(|(i, &p)| describe(instance, p, i, compute_queue_family(instance, p)))
            .collect();
        if infos.is_empty() {
            return err("no Vulkan devices found");
        }
        let usable = |i: &usize| infos[*i].timestamp_valid_bits > 0;
        let picked = match selector {
            DeviceSelector::Index(i) => (*i < infos.len()).then_some(*i),
            DeviceSelector::NameContains(s) => {
                let s = s.to_lowercase();
                (0..infos.len()).filter(usable).find(|&i| infos[i].name.to_lowercase().contains(&s))
            }
            DeviceSelector::Auto => (0..infos.len())
                .filter(usable)
                .find(|&i| infos[i].device_type == "discrete")
                .or_else(|| (0..infos.len()).filter(usable).find(|&i| infos[i].device_type == "integrated"))
                .or_else(|| (0..infos.len()).find(usable)),
        };
        let Some(i) = picked else {
            return err("no matching Vulkan device with a compute queue and timestamps");
        };
        let phys = phys_all[i];
        let Some(qf) = compute_queue_family(instance, phys) else {
            return err("selected device has no compute queue with timestamp support");
        };

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let has = |n: &str| infos[i].extensions.iter().any(|e| e == n);
        let want_coop = has("VK_KHR_cooperative_matrix") && has("VK_KHR_shader_float16_int8") && has("VK_KHR_16bit_storage");
        let mut dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default().shader_integer_dot_product(true);
        let mut coop_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut f16_features = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true).uniform_and_storage_buffer16_bit_access(true);
        let mut mem_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true).vulkan_memory_model_device_scope(true);
        let mut exec_info = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        let want_exec = has("VK_KHR_pipeline_executable_properties");
        let want_sgsize = has("VK_EXT_subgroup_size_control");
        let mut sg_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true).compute_full_subgroups(true);
        let mut ext_names: Vec<*const std::ffi::c_char> = Vec::new();
        if infos[i].int_dot {
            ext_names.push(c"VK_KHR_shader_integer_dot_product".as_ptr());
        }
        if want_coop {
            ext_names.push(c"VK_KHR_cooperative_matrix".as_ptr());
            ext_names.push(c"VK_KHR_shader_float16_int8".as_ptr());
            ext_names.push(c"VK_KHR_16bit_storage".as_ptr());
            ext_names.push(c"VK_KHR_vulkan_memory_model".as_ptr());
        }
        if want_exec {
            ext_names.push(c"VK_KHR_pipeline_executable_properties".as_ptr());
        }
        if want_sgsize {
            ext_names.push(c"VK_EXT_subgroup_size_control".as_ptr());
        }
        let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&ext_names);
        if want_exec {
            dci = dci.push_next(&mut exec_info);
        }
        if want_sgsize {
            dci = dci.push_next(&mut sg_control);
        }
        if infos[i].int_dot {
            dci = dci.push_next(&mut dot_features);
        }
        if want_coop {
            dci = dci.push_next(&mut coop_features).push_next(&mut f16_features).push_next(&mut storage16).push_next(&mut mem_model);
        }
        infos[i].coopmat = want_coop;        let device = unsafe { instance.create_device(phys, &dci, None) }?;
        let queue = unsafe { device.get_device_queue(qf, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                unsafe { device.destroy_device(None) };
                return Err(e.into());
            }
        };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
        Ok(Self {
            _entry: entry,
            instance: instance.clone(),
            phys,
            device,
            queue,
            cmd_pool,
            mem_props,
            info: infos[i].clone(),
        })
    }

    /// Human-readable memory heaps and types, and the type chosen for storage buffers.
    pub fn memory_report(&self) -> String {
        let mp = &self.mem_props;
        let mut out = String::new();
        for (i, h) in mp.memory_heaps[..mp.memory_heap_count as usize].iter().enumerate() {
            out += &format!("heap {i}: {:.2} GiB flags {}\n", h.size as f64 / (1u64 << 30) as f64, h.flags.as_raw());
        }
        for (i, t) in mp.memory_types[..mp.memory_type_count as usize].iter().enumerate() {
            out += &format!("type {i}: heap {} flags {:#06x} (1=device_local 2=host_visible 4=coherent 8=cached)\n", t.heap_index, t.property_flags.as_raw());
        }
        out
    }

    /// Cooperative-matrix configurations the driver supports, one line each.
    pub fn cooperative_matrix_report(&self) -> Vec<String> {
        if !self.info.extensions.iter().any(|e| e == "VK_KHR_cooperative_matrix") {
            return Vec::new();
        }
        let loader = ash::khr::cooperative_matrix::Instance::new(&self._entry, &self.instance);
        let props = unsafe { loader.get_physical_device_cooperative_matrix_properties(self.phys) }.unwrap_or_default();
        props
            .iter()
            .map(|p| format!("M={} N={} K={} A={} B={} C={} R={} scope={} sat={}", p.m_size, p.n_size, p.k_size, p.a_type.as_raw(), p.b_type.as_raw(), p.c_type.as_raw(), p.result_type.as_raw(), p.scope.as_raw(), p.saturating_accumulation))
            .collect()
    }

    /// Driver statistics (register counts, LDS, ...) of a kernel created while
    /// LEV_PIPELINE_INFO was set; with dump_dir the compiled ISA text is
    /// written there as <name>.isa.txt.
    pub fn kernel_report(&self, kernel: &Kernel, name: &str, dump_dir: Option<&Path>) -> Result<Vec<String>, VkError> {
        if !self.info.extensions.iter().any(|e| e == "VK_KHR_pipeline_executable_properties") {
            return err("VK_KHR_pipeline_executable_properties is not available");
        }
        let loader = ash::khr::pipeline_executable_properties::Device::new(&self.instance, &self.device);
        let pipeline = kernel.raw_pipeline();
        let mut out = Vec::new();
        unsafe {
            let exes = loader.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline))?;
            for (i, e) in exes.iter().enumerate() {
                let ename = CStr::from_ptr(e.name.as_ptr()).to_string_lossy().into_owned();
                out.push(format!("executable {i}: {ename} (subgroup size {})", e.subgroup_size));
                let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i as u32);
                for st in loader.get_pipeline_executable_statistics(&info)? {
                    let n = CStr::from_ptr(st.name.as_ptr()).to_string_lossy().into_owned();
                    let v = match st.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (st.value.b32 != 0).to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => st.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => st.value.u64.to_string(),
                        _ => st.value.f64.to_string(),
                    };
                    out.push(format!("  {n} = {v}"));
                }
                if let Some(dir) = dump_dir {
                    // Two phases: sizes first, then the text into caller buffers.
                    let fp = loader.fp();
                    let mut count = 0u32;
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(self.device.handle(), &info, &mut count, std::ptr::null_mut());
                    let mut irs = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(self.device.handle(), &info, &mut count, irs.as_mut_ptr());
                    let mut bufs: Vec<Vec<u8>> = irs.iter().map(|ir| vec![0u8; ir.data_size]).collect();
                    for (ir, buf) in irs.iter_mut().zip(bufs.iter_mut()) {
                        ir.p_data = buf.as_mut_ptr() as *mut c_void;
                    }
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(self.device.handle(), &info, &mut count, irs.as_mut_ptr());
                    for (ir, buf) in irs.iter().zip(&bufs) {
                        let irn = CStr::from_ptr(ir.name.as_ptr()).to_string_lossy().into_owned();
                        if ir.is_text == vk::TRUE {
                            let _ = fs::create_dir_all(dir);
                            let file = dir.join(format!("{name}.{}.txt", irn.replace([' ', '/'], "_")));
                            let _ = fs::write(&file, buf);
                            out.push(format!("  wrote {} ({} bytes)", file.display(), buf.len()));
                        }
                    }
                }            }
        }
        let _ = name;
        Ok(out)
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn find_memory_type(&self, type_bits: u32, required: vk::MemoryPropertyFlags, preferred: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.mem_props.memory_types[..self.mem_props.memory_type_count as usize];
        let ok = |i: usize, flags: vk::MemoryPropertyFlags| {
            type_bits & (1 << i) != 0 && types[i].property_flags.contains(flags)
        };
        (0..types.len())
            .find(|&i| ok(i, required | preferred))
            .or_else(|| (0..types.len()).find(|&i| ok(i, required)))
            .map(|i| i as u32)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance() -> Result<(Entry, Instance), VkError> {
    let entry = unsafe { Entry::load() }.map_err(|e| VkError(format!("cannot load the Vulkan loader: {e}")))?;
    let app = vk::ApplicationInfo::default()
        .application_name(c"leviathan")
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&ci, None) }?;
    Ok((entry, instance))
}

/// Host-visible, host-coherent storage buffer, mapped for its whole lifetime.
struct GpuBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut c_void,
    size: u64,
}

impl GpuBuffer {
    fn new(ctx: &VulkanContext, size: u64) -> Result<Self, VkError> {
        let device = ctx.device.clone();
        let bci = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&bci, None) }?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let Some(mt) = ctx.find_memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            unsafe { device.destroy_buffer(buffer, None) };
            return err("no host-visible coherent memory type");
        };
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
        let memory = match unsafe { device.allocate_memory(&ai, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(e.into());
            }
        };
        let bound = unsafe { device.bind_buffer_memory(buffer, memory, 0) };
        let mapped = bound.and_then(|_| unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) });
        match mapped {
            Ok(ptr) => Ok(Self { device, buffer, memory, ptr, size }),
            Err(e) => {
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_buffer(buffer, None);
                }
                Err(e.into())
            }
        }
    }

    /// Device-local buffer that the host cannot map; filled through staging
    /// copies. Kept out of host-visible memory because the GPU reads such
    /// mappings through a slower, uncached path on this class of device.
    fn new_device(ctx: &VulkanContext, size: u64) -> Result<Self, VkError> {
        let device = ctx.device.clone();
        let bci = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&bci, None) }?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let types = &ctx.mem_props.memory_types[..ctx.mem_props.memory_type_count as usize];
        let mt = (0..types.len())
            .find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !types[i].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .or_else(|| ctx.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty()).map(|i| i as usize));
        let Some(mt) = mt else {
            unsafe { device.destroy_buffer(buffer, None) };
            return err("no device-local memory type");
        };
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt as u32);
        let memory = match unsafe { device.allocate_memory(&ai, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(e.into());
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(e.into());
        }
        Ok(Self { device, buffer, memory, ptr: std::ptr::null_mut(), size })
    }

    fn write(&self, bytes: &[u8]) {
        assert!(!self.ptr.is_null(), "buffer is not host-mappable");
        assert!(bytes.len() as u64 <= self.size);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr as *mut u8, bytes.len()) };
    }

    fn read_f32(&self, count: usize) -> Vec<f32> {
        assert!(!self.ptr.is_null(), "buffer is not host-mappable");
        assert!((count * 4) as u64 <= self.size);
        let mut out = vec![0f32; count];
        unsafe { std::ptr::copy_nonoverlapping(self.ptr as *const f32, out.as_mut_ptr(), count) };
        out
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Per-run Vulkan objects; null handles are legal to destroy.
struct RunObjects<'a> {
    device: &'a Device,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
    query_pool: vk::QueryPool,
    fence: vk::Fence,
}

impl Drop for RunObjects<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipe_layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Debug)]
pub struct RunResult {
    pub y: Vec<f32>,
    /// GPU time of each timed run, microseconds (device timestamps).
    pub gpu_us: Vec<f64>,
}

impl RunResult {
    pub fn median_us(&self) -> f64 {
        let mut v = self.gpu_us.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
}

impl VulkanContext {
    /// Runs the Q4_K matvec `y = W x` for `rows x cols` weights in Q4_K blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn run_q4k_matvec(
        &self,
        spirv: &[u32],
        schedule: &Q4kSchedule,
        weights: &[u8],
        rows: u32,
        cols: u32,
        x: &[f32],
        warmups: usize,
        timed: usize,
    ) -> Result<RunResult, VkError> {
        self.run_matvec(spirv, group_count(schedule, rows), Q4K_BLOCK_BYTES, weights, rows, cols, x, warmups, timed)
    }

    /// Runs a block-quantized matvec `y = W x` with `block_bytes` bytes per
    /// 256-element block and `groups` workgroups (bindings: weights, x, y; push
    /// constants: rows, blocks per row). The weights buffer is padded by 16
    /// bytes so kernels may read a second word of an unaligned load.
    /// `warmups` untimed and `timed` timed submissions; the output of the last
    /// run is returned. Timing uses device timestamps around the dispatch only.
    #[allow(clippy::too_many_arguments)]
    pub fn run_matvec(
        &self,
        spirv: &[u32],
        groups: u32,
        block_bytes: usize,
        weights: &[u8],
        rows: u32,
        cols: u32,
        x: &[f32],
        warmups: usize,
        timed: usize,
    ) -> Result<RunResult, VkError> {
        if timed == 0 {
            return err("timed must be at least 1");
        }
        if cols == 0 || cols as usize % QK_K != 0 {
            return err("cols must be a positive multiple of 256");
        }
        let blocks_per_row = cols as usize / QK_K;
        let expected = rows as usize * blocks_per_row * block_bytes;
        if weights.len() != expected || x.len() != cols as usize || rows == 0 {
            return err("weights or input length does not match rows x cols");
        }
        if groups > self.info.max_group_count_x {
            return err(format!("{groups} workgroups exceed the device limit {}", self.info.max_group_count_x));
        }

        let w_buf = GpuBuffer::new(self, weights.len() as u64 + 16)?;
        let x_buf = GpuBuffer::new(self, (x.len() * 4) as u64)?;
        let y_buf = GpuBuffer::new(self, rows as u64 * 4)?;
        w_buf.write(weights);
        x_buf.write(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) });

        let d = &self.device;
        let mut o = RunObjects {
            device: d,
            module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipe_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            desc_pool: vk::DescriptorPool::null(),
            query_pool: vk::QueryPool::null(),
            fence: vk::Fence::null(),
        };

        unsafe {
            o.module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;
            let bindings: Vec<_> = (0..3)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            o.set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)];
            let layouts = [o.set_layout];
            o.pipe_layout = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(o.module)
                .name(c"main");
            let pci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(o.pipe_layout)];
            o.pipeline = match d.create_compute_pipelines(vk::PipelineCache::null(), &pci, None) {
                Ok(p) => p[0],
                Err((_, e)) => return Err(e.into()),
            };

            let pool_sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3)];
            o.desc_pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None)?;
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(o.desc_pool).set_layouts(&layouts))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [&w_buf, &x_buf, &y_buf]
                .iter()
                .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);

            o.query_pool = d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                None,
            )?;
            o.fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            let mut push = [0u8; 8];
            push[..4].copy_from_slice(&rows.to_ne_bytes());
            push[4..].copy_from_slice(&(blocks_per_row as u32).to_ne_bytes());
            let mask = if self.info.timestamp_valid_bits >= 64 { u64::MAX } else { (1u64 << self.info.timestamp_valid_bits) - 1 };

            let mut gpu_us = Vec::with_capacity(timed);
            let result = (|| -> Result<(), VkError> {
                for run in 0..warmups + timed {
                    d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                    d.cmd_reset_query_pool(cmd, o.query_pool, 0, 2);
                    d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, o.query_pool, 0);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, o.pipeline);
                    d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, o.pipe_layout, 0, &[set], &[]);
                    d.cmd_push_constants(cmd, o.pipe_layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                    d.cmd_dispatch(cmd, groups, 1, 1);
                    d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, o.query_pool, 1);
                    d.end_command_buffer(cmd)?;
                    let cmds = [cmd];
                    d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], o.fence)?;
                    d.wait_for_fences(&[o.fence], true, u64::MAX)?;
                    d.reset_fences(&[o.fence])?;
                    let mut ts = [0u64; 2];
                    d.get_query_pool_results(o.query_pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
                    if run >= warmups {
                        let ticks = (ts[1] & mask).wrapping_sub(ts[0] & mask) & mask;
                        gpu_us.push(ticks as f64 * self.info.timestamp_period_ns as f64 / 1000.0);
                    }
                    d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
                }
                Ok(())
            })();
            d.free_command_buffers(self.cmd_pool, &[cmd]);
            result?;

            Ok(RunResult { y: y_buf.read_f32(rows as usize), gpu_us })
        }
    }
}

/// Locates `glslc`: `LEV_GLSLC`, then the Vulkan SDK, then `PATH`.
pub fn glslc_path() -> PathBuf {
    if let Ok(p) = std::env::var("LEV_GLSLC") {
        return PathBuf::from(p);
    }
    let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        for dir in ["Bin", "bin"] {
            let p = Path::new(&sdk).join(dir).join(exe);
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from(exe)
}

fn spirv_words(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return None;
    }
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (words[0] == 0x0723_0203).then_some(words)
}

/// Compiles GLSL to SPIR-V with `glslc`, content-addressed by the source
/// digest: a hit is reused, a corrupt entry is deleted and rebuilt, and a
/// failed compile leaves no files behind (only the returned error text).
pub fn compile_glsl_cached(src: &str, cache_dir: &Path) -> Result<(Vec<u32>, bool), VkError> {
    let digest = source_digest(src);
    let spv = cache_dir.join(format!("{digest}.spv"));
    if let Ok(bytes) = fs::read(&spv) {
        match spirv_words(&bytes) {
            Some(words) => return Ok((words, true)),
            None => {
                let _ = fs::remove_file(&spv);
            }
        }
    }
    fs::create_dir_all(cache_dir)?;
    let tmp_src = cache_dir.join(format!("{digest}.comp.tmp"));
    let tmp_out = cache_dir.join(format!("{digest}.spv.tmp"));
    fs::write(&tmp_src, src)?;
    let output = Command::new(glslc_path())
        .args([if src.contains("GL_KHR_cooperative_matrix") { "--target-env=vulkan1.3" } else { "--target-env=vulkan1.1" }, "-fshader-stage=compute"])
        .arg(&tmp_src)
        .arg("-o")
        .arg(&tmp_out)
        .output();
    let cleanup = || {
        let _ = fs::remove_file(&tmp_src);
        let _ = fs::remove_file(&tmp_out);
    };
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            cleanup();
            return err(format!("cannot run glslc ({e}); set VULKAN_SDK or LEV_GLSLC"));
        }
    };
    if !output.status.success() {
        let mut log = String::from_utf8_lossy(&output.stderr).into_owned();
        log.truncate(MAX_LOG_BYTES);
        cleanup();
        return err(format!("glslc failed: {log}"));
    }
    let bytes = fs::read(&tmp_out)?;
    let Some(words) = spirv_words(&bytes) else {
        cleanup();
        return err("glslc produced invalid SPIR-V");
    };
    fs::rename(&tmp_out, &spv)?;
    let _ = fs::remove_file(&tmp_src);
    Ok((words, false))
}
