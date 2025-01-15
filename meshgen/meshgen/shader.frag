#version 430 core

uniform sampler2D img;
uniform vec2 scale;

in vec2 texCoord;
out vec4 FragColor;

uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

float hm(uint m) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

vec3 color(float i_f) {
	uint i = floatBitsToUint(i_f + 1);

	uint r = hash(i);
	uint g = hash(r);
	uint b = hash(g);

	return vec3(hm(r), hm(g), hm(b));
}

const vec3 light_normal = vec3(.5, .0, .866);

void main()
{
	vec2 tex_offset = 2.0 / textureSize(img, 0);
	
	vec4 u = texture(img, texCoord);

	float u_px = texture(img, texCoord + vec2(tex_offset.x, 0)).r;
	float u_py = texture(img, texCoord + vec2(0, tex_offset.y)).r;
	float u_nx = texture(img, texCoord - vec2(tex_offset.x, 0)).r;
	float u_ny = texture(img, texCoord - vec2(0, tex_offset.y)).r;
	
	vec2 h = 2 * tex_offset * scale;
	vec3 dx = vec3(h.x, 0, u_px - u_nx);
	vec3 dy = vec3(0, h.y, u_py - u_ny);
	vec3 n = normalize(cross(dx, dy));

	//FragColor = vec4(color(u.y) * .5 * (dot(n, light_normal) + 1), 1);
	FragColor = vec4(n, 0);
}