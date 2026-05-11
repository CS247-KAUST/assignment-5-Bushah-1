#version 450

in vec2 texCoord;

uniform sampler2D txtr;
uniform vec4 vertexColor;

// colormap mode: 0=off, 1=rainbow, 2=cool-warm
uniform int colormapMode;
// blend factor between grayscale (0.0) and color-mapped (1.0)
uniform float blendFactor;

out vec4 fragColor;

// Rainbow colormap: blue -> cyan -> green -> yellow -> red
vec3 rainbow(float t) {
    vec3 c;
    if (t < 0.25) {
        c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t / 0.25);
    } else if (t < 0.5) {
        c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) / 0.25);
    } else {
        c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) / 0.25);
    }
    return c;
}

// Cool-warm colormap: blue -> white -> red
vec3 coolWarm(float t) {
    if (t < 0.5) {
        return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), t / 0.5);
    } else {
        return mix(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), (t - 0.5) / 0.5);
    }
}

void main() {
    vec4 texColor = texture(txtr, texCoord);
    float scalar = texColor.r;

    if (colormapMode == 0) {
        // original behavior: use vertexColor for overlays or grayscale texture
        fragColor = vertexColor + texColor;
    } else {
        // apply colormap
        vec3 gray = vec3(scalar);
        vec3 mapped;
        if (colormapMode == 1) {
            mapped = rainbow(scalar);
        } else {
            mapped = coolWarm(scalar);
        }
        vec3 blended = mix(gray, mapped, blendFactor);
        fragColor = vec4(blended, 1.0);
    }
}
