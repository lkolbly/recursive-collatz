@group(0)
@binding(0)
var<storage, read_write> v_indices_out: array<u32>; // output

struct Params {
    start: u32;
};
@group(0)
@binding(1)
var<uniform> start_index: Params;

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //let start_idx: u32 = 1u;
    v_indices_out[global_id.x] = global_id.x + start_index.start;
}
