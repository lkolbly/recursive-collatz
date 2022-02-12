use std::borrow::Cow;
use std::collections::HashSet;
use wgpu::util::DeviceExt;
use wgpu::Buffer;

// Indicates a u32 overflow in an intermediate Collatz value
const OVERFLOW: u32 = 0xffffffff;

async fn run() {
    let steps = execute_gpu(1_000_000_000).await;

    let fixed_points: Vec<String> = steps
        .iter()
        .map(|&n| match n {
            OVERFLOW => "OVERFLOW".to_string(),
            _ => n.to_string(),
        })
        .collect();

    println!("Steps: [{}]", fixed_points.join(", "));
}

async fn execute_gpu(count: u32) -> HashSet<u32> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    let info = adapter.get_info();
    println!("Using {} ({:?})", info.name, info.backend);

    execute_gpu_inner(&device, &queue, count).await
}

async fn execute_gpu_inner(device: &wgpu::Device, queue: &wgpu::Queue, count: u32) -> HashSet<u32> {
    // Loads the shader from WGSL
    println!("Loading collatz module...");
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    println!("Loading fill module...");
    let fill_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("fill.wgsl"))),
    });
    println!("Modules loaded");

    // Gets the size in bytes of the buffer.
    let block_size = 65535;
    let slice_size = block_size as usize * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer 1"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Instantiates a second buffer without data
    // This buffer ping-pongs with the buffer above
    let storage_buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer 2"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let start_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Start Index Buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &fill_module,
        entry_point: "main",
    });

    let mut results: HashSet<u32> = HashSet::new();

    let mut start = 0;
    while start < count {
        let fixed_points = compute_range(
            &device,
            &queue,
            &fill_pipeline,
            &compute_pipeline,
            &storage_buffer,
            &storage_buffer2,
            &staging_buffer,
            &start_index_buffer,
            size,
            start + 1,
            block_size,
        )
        .await;

        fixed_points.iter().for_each(|&n| {
            results.insert(n);
        });

        start += block_size;
    }
    println!("{:?}", results);

    results
}

async fn compute_range(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fill_pipeline: &wgpu::ComputePipeline,
    compute_pipeline: &wgpu::ComputePipeline,
    storage_buffer: &wgpu::Buffer,
    storage_buffer2: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    start_index_buffer: &wgpu::Buffer,
    size: u64,  // Size in bytes of each buffer
    start: u32, // First element to process
    count: u32, // Number of elements to process
) -> Vec<u32> {
    // Fill the buffer with initial values
    {
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        queue.write_buffer(&start_index_buffer, 0, bytemuck::cast_slice(&[start]));

        initialize_buffer(
            &device,
            &fill_pipeline,
            &mut encoder,
            &storage_buffer,
            &start_index_buffer,
            count,
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    let itercount = 100;

    let timer = std::time::Instant::now();
    for idx in 0..itercount {
        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            for _ in 0..9 {
                compute_collatz(
                    &device,
                    &compute_pipeline,
                    &mut encoder,
                    &storage_buffer,
                    &storage_buffer2,
                    count,
                );
                compute_collatz(
                    &device,
                    &compute_pipeline,
                    &mut encoder,
                    &storage_buffer2,
                    &storage_buffer,
                    count,
                );
            }

            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        let buf1 = retrieve_buffer(&device, &queue, &storage_buffer, &staging_buffer, size)
            .await
            .unwrap();
        let buf2 = retrieve_buffer(&device, &queue, &storage_buffer2, &staging_buffer, size)
            .await
            .unwrap();
        if buf1 == buf2 {
            println!(
                "{}: Converged after {} iters, elapsed = {}s",
                start,
                idx,
                timer.elapsed().as_secs_f32()
            );
            break;
        }
    }

    retrieve_buffer(&device, &queue, &storage_buffer, &staging_buffer, size)
        .await
        .unwrap()
}

/// Initializes a given buffer to contain the numbers from start to stop, inclusive
fn initialize_buffer(
    device: &wgpu::Device,
    fill_pipeline: &wgpu::ComputePipeline,
    encoder: &mut wgpu::CommandEncoder,
    dest: &Buffer,
    start_uniform_buffer: &Buffer,
    count: u32,
) {
    let bind_group_layout = fill_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: start_uniform_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&fill_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("fill buffer");
        cpass.dispatch(count as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
}

/// Runs the collatz shader, moving the count numbers data from the source
/// buffer to the dest buffer.
/// Both source and dest must be large enough to accomodate count numbers.
fn compute_collatz(
    device: &wgpu::Device,
    compute_pipeline: &wgpu::ComputePipeline,
    encoder: &mut wgpu::CommandEncoder,
    source: &Buffer,
    dest: &Buffer,
    count: u32,
) {
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dest.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(count, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
}

async fn retrieve_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &Buffer,
    dest: &Buffer,
    size: u64,
) -> Option<Vec<u32>> {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&source, 0, &dest, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = dest.slice(..);
    // Gets the future representing when `dest` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        dest.unmap(); // Unmaps buffer from memory
                      // If you are familiar with C++ these 2 lines can be thought of similarly to:
                      //   delete myPointer;
                      //   myPointer = NULL;
                      // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
