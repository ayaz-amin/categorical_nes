pub struct RNG
{
    seed: u32,
    pos: u32
}

impl RNG
{
    pub fn new(seed: u32) -> Self
    {
        Self {seed, pos: 1}
    }

    pub fn sample(&mut self) -> f32
    {
        const NOISE1: u32 = 0x68E31DA4;
        const NOISE2: u32 = 0xB5297A4D;
        const NOISE3: u32 = 0x1B56C4E9;

        let mut mangled = self.pos as u32;
        mangled = mangled.wrapping_mul(NOISE1);
        mangled = mangled.wrapping_add(self.seed);
        mangled ^= mangled >> 8;
        mangled = mangled.wrapping_add(NOISE2);
        mangled ^= mangled << 8;
        mangled = mangled.wrapping_mul(NOISE3);
        mangled ^= mangled >> 8;
        self.pos += 1;
        (mangled as f32) / (u32::MAX as f32)
    }
}