//! Recorded compute programs: persistent buffers, pipelines with N storage
//! bindings and push constants, and a command buffer recorded once and
//! replayed. Per-token values travel through a buffer, not push constants.

use ash::{vk, Device};

use crate::{err, GpuBuffer, VkError, VulkanContext};

/// Host-visible device buffer, mapped for its lifetime.
pub struct Buf(GpuBuffer);

impl Buf {
    pub fn size(&self) -> u64 {
        self.0.size
    }

    /// Whether the host can read and write this buffer directly.
    pub fn is_mapped(&self) -> bool {
        !self.0.ptr.is_null()
    }

    pub fn write_bytes(&self, offset: usize, bytes: &[u8]) {
        assert!(self.is_mapped(), "buffer is device-local; use VulkanContext::upload");
        assert!((offset + bytes.len()) as u64 <= self.0.size);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), (self.0.ptr as *mut u8).add(offset), bytes.len()) };
    }

    pub fn write_f32(&self, offset_floats: usize, v: &[f32]) {
        assert!(((offset_floats + v.len()) * 4) as u64 <= self.0.size);
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), (self.0.ptr as *mut f32).add(offset_floats), v.len()) };
    }

    pub fn write_u32(&self, offset_words: usize, v: &[u32]) {
        assert!(((offset_words + v.len()) * 4) as u64 <= self.0.size);
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), (self.0.ptr as *mut u32).add(offset_words), v.len()) };
    }

    pub fn read_f32(&self, offset_floats: usize, count: usize) -> Vec<f32> {
        assert!(((offset_floats + count) * 4) as u64 <= self.0.size);
        let mut out = vec![0f32; count];
        unsafe { std::ptr::copy_nonoverlapping((self.0.ptr as *const f32).add(offset_floats), out.as_mut_ptr(), count) };
        out
    }

    /// Copies out.len() bytes starting at byte offset out of a mapped buffer.
    pub fn read_bytes(&self, offset: usize, out: &mut [u8]) {
        assert!(self.is_mapped(), "buffer is device-local");
        assert!((offset + out.len()) as u64 <= self.0.size);
        unsafe { std::ptr::copy_nonoverlapping((self.0.ptr as *const u8).add(offset), out.as_mut_ptr(), out.len()) };
    }

    pub fn zero(&self) {
        assert!(self.is_mapped(), "buffer is device-local; use VulkanContext::zero_buffer");
        unsafe { std::ptr::write_bytes(self.0.ptr as *mut u8, 0, self.0.size as usize) };
    }
}

pub struct Kernel {
    device: Device,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    n_bindings: usize,
    push_size: u32,
}

impl Kernel {
    pub(crate) fn raw_pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipe_layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

/// One dispatch of a program: a kernel, its bound buffer ranges
/// `(buffer, byte offset, byte range)`, push constants and the group count.
pub struct Step<'a> {
    pub kernel: &'a Kernel,
    pub bindings: Vec<(&'a Buf, u64, u64)>,
    pub push: Vec<u8>,
    pub groups: u32,
}

pub struct Program {
    device: Device,
    queue: vk::Queue,
    pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    fence: vk::Fence,
    period_ns: f64,
    pub steps: usize,
    profiled: bool,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            if self.cmd != vk::CommandBuffer::null() {
                self.device.free_command_buffers(self.cmd_pool, &[self.cmd]);
            }
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

impl Program {
    /// Submits the recorded program and waits; returns the GPU time in
    /// microseconds measured by device timestamps around the whole program.
    /// Per-step GPU time in microseconds (needs a program built with `profile`).
    pub fn run_profile(&self) -> Result<Vec<f64>, VkError> {
        if !self.profiled {
            return err("program was not built with profiling");
        }
        unsafe {
            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
            let mut ts = vec![0u64; self.steps + 2];
            self.device.get_query_pool_results(self.query_pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            Ok((1..=self.steps).map(|i| ts[i].wrapping_sub(ts[i - 1]) as f64 * self.period_ns / 1000.0).collect())
        }
    }

    pub fn run(&self) -> Result<f64, VkError> {
        unsafe {
            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
            let mut ts = [0u64; 2];
            self.device.get_query_pool_results(self.query_pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            Ok(ts[if self.profiled { self.steps + 1 } else { 1 }].wrapping_sub(ts[0]) as f64 * self.period_ns / 1000.0)
        }
    }
}

impl VulkanContext {
    pub fn create_buffer(&self, size: u64) -> Result<Buf, VkError> {
        Ok(Buf(GpuBuffer::new(self, size)?))
    }

    /// Device-local (not host-mappable) buffer; see `upload` and `zero_buffer`.
    pub fn create_device_buffer(&self, size: u64) -> Result<Buf, VkError> {
        Ok(Buf(GpuBuffer::new_device(self, size)?))
    }

    fn submit_once(&self, record: impl FnOnce(&Device, vk::CommandBuffer)) -> Result<(), VkError> {
        let d = &self.device;
        unsafe {
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            let result = (|| -> Result<(), VkError> {
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                record(d, cmd);
                d.end_command_buffer(cmd)?;
                let cmds = [cmd];
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], vk::Fence::null())?;
                d.queue_wait_idle(self.queue)?;
                Ok(())
            })();
            d.free_command_buffers(self.cmd_pool, &[cmd]);
            result
        }
    }

    /// Copies `data` into `dst` at `offset` through a staging buffer.
    pub fn upload(&self, dst: &Buf, offset: u64, data: &[u8]) -> Result<(), VkError> {
        if offset + data.len() as u64 > dst.size() {
            return err("upload outside the buffer");
        }
        const CHUNK: usize = 64 << 20;
        let staging = GpuBuffer::new(self, CHUNK.min(data.len()).max(4) as u64)?;
        let mut done = 0usize;
        while done < data.len() {
            let n = CHUNK.min(data.len() - done);
            staging.write(&data[done..done + n]);
            let (src, dstb, dst_off) = (staging.buffer, dst.0.buffer, offset + done as u64);
            self.submit_once(|d, cmd| unsafe {
                d.cmd_copy_buffer(cmd, src, dstb, &[vk::BufferCopy::default().src_offset(0).dst_offset(dst_off).size(n as u64)]);
            })?;
            done += n;
        }
        Ok(())
    }

    /// Copies len bytes at offset of any buffer back to the host.
    pub fn download(&self, src: &Buf, offset: u64, len: u64) -> Result<Vec<u8>, VkError> {
        if offset + len > src.size() {
            return err("download outside the buffer");
        }
        let staging = GpuBuffer::new(self, len.max(4))?;
        let (sb, db) = (src.0.buffer, staging.buffer);
        self.submit_once(|d, cmd| unsafe {
            d.cmd_copy_buffer(cmd, sb, db, &[vk::BufferCopy::default().src_offset(offset).dst_offset(0).size(len)]);
        })?;
        let mut out = vec![0u8; len as usize];
        unsafe { std::ptr::copy_nonoverlapping(staging.ptr as *const u8, out.as_mut_ptr(), len as usize) };
        Ok(out)
    }

    /// Fills a device-local buffer with zeros.
    pub fn zero_buffer(&self, b: &Buf) -> Result<(), VkError> {
        let handle = b.0.buffer;
        self.submit_once(|d, cmd| unsafe { d.cmd_fill_buffer(cmd, handle, 0, vk::WHOLE_SIZE, 0) })
    }

    pub fn create_kernel(&self, spirv: &[u32], n_bindings: usize, push_size: u32) -> Result<Kernel, VkError> {
        self.create_kernel_sub(spirv, n_bindings, push_size, None)
    }

    /// Like create_kernel, optionally forcing the subgroup size (needs
    /// VK_EXT_subgroup_size_control; the size must be within the device range).
    pub fn create_kernel_sub(&self, spirv: &[u32], n_bindings: usize, push_size: u32, subgroup_size: Option<u32>) -> Result<Kernel, VkError> {
        let d = &self.device;
        let mut k = Kernel {
            device: d.clone(),
            module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipe_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            n_bindings,
            push_size,
        };
        unsafe {
            k.module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;
            let bindings: Vec<_> = (0..n_bindings as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            k.set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(push_size.max(4))];
            let layouts = [k.set_layout];
            let mut plci = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            if push_size > 0 {
                plci = plci.push_constant_ranges(&ranges);
            }
            k.pipe_layout = d.create_pipeline_layout(&plci, None)?;
            let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(subgroup_size.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(k.module)
                .name(c"main");
            if subgroup_size.is_some() {
                stage = stage.push_next(&mut req);
            }
            let mut pci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(k.pipe_layout);
            if std::env::var_os("LEV_PIPELINE_INFO").is_some() {
                pci = pci.flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR);
            }
            k.pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[pci], None)
                .map_err(|(_, e)| VkError(format!("pipeline creation failed: {e:?}")))?[0];
        }
        Ok(k)
    }

    /// Records `steps` into one command buffer with a full compute barrier
    /// after each dispatch. Buffers and kernels must outlive the program.
    pub fn build_program(&self, steps: &[Step<'_>], profile: bool) -> Result<Program, VkError> {
        if steps.is_empty() {
            return err("empty program");
        }
        let d = &self.device;
        let total_desc: usize = steps.iter().map(|s| s.bindings.len()).sum();
        for (i, s) in steps.iter().enumerate() {
            if s.bindings.len() != s.kernel.n_bindings || s.push.len() as u32 != s.kernel.push_size {
                return err(format!("step {i}: binding or push size mismatch"));
            }
            if s.groups == 0 || s.groups > self.info.max_group_count_x {
                return err(format!("step {i}: invalid group count {}", s.groups));
            }
            for (b, off, range) in &s.bindings {
                if off + range > b.size() || off % 256 != 0 {
                    return err(format!("step {i}: binding range outside the buffer or offset not 256-aligned"));
                }
            }
        }
        unsafe {
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(total_desc.max(1) as u32)];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(steps.len() as u32).pool_sizes(&sizes), None)?;
            let mut prog = Program {
                device: d.clone(),
                queue: self.queue,
                pool,
                cmd_pool: self.cmd_pool,
                cmd: vk::CommandBuffer::null(),
                query_pool: vk::QueryPool::null(),
                fence: vk::Fence::null(),
                period_ns: self.info.timestamp_period_ns as f64,
                steps: steps.len(),
                profiled: profile,
            };
            let layouts: Vec<_> = steps.iter().map(|s| s.kernel.set_layout).collect();
            let sets = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            for (s, set) in steps.iter().zip(&sets) {
                let infos: Vec<[vk::DescriptorBufferInfo; 1]> = s
                    .bindings
                    .iter()
                    .map(|(b, off, range)| [vk::DescriptorBufferInfo::default().buffer(b.0.buffer).offset(*off).range(*range)])
                    .collect();
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                d.update_descriptor_sets(&writes, &[]);
            }
            prog.cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            prog.query_pool = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(if profile { steps.len() as u32 + 2 } else { 2 }), None)?;
            prog.fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let cmd = prog.cmd;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cmd, prog.query_pool, 0, if profile { steps.len() as u32 + 2 } else { 2 });
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, prog.query_pool, 0);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            for (si, (s, set)) in steps.iter().zip(&sets).enumerate() {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, s.kernel.pipeline);
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, s.kernel.pipe_layout, 0, &[*set], &[]);
                if s.kernel.push_size > 0 {
                    d.cmd_push_constants(cmd, s.kernel.pipe_layout, vk::ShaderStageFlags::COMPUTE, 0, &s.push);
                }
                d.cmd_dispatch(cmd, s.groups, 1, 1);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                if profile {
                    d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, prog.query_pool, si as u32 + 1);
                }
            }
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, prog.query_pool, if profile { steps.len() as u32 + 1 } else { 1 });
            d.end_command_buffer(cmd)?;
            Ok(prog)
        }
    }
}
