@group(0)
@binding(0)
var<storage, read_write> v_indices_in: array<u32>; // input

@group(0)
@binding(1)
var<storage, read_write> v_indices_out: array<u32>; // output

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        }
        else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }

            n = 3u * n + 1u;
        }
        i = i + 1u;

        // Force the whole warp to synchronize here
        workgroupBarrier();
    }
    return i;
}

@stage(compute)
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices_out[global_id.x] = collatz_iterations(v_indices_in[global_id.x]);
}
