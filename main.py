from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
from ursina.shader import Shader as UrsinaShader
from panda3d.core import BitMask32, TextureStage, Vec4, Mat4, Texture, Shader, Vec2
import random, math, time as _time

app = Ursina()

# Kare Arttırıcı çıkışı 1080p; seçilen profile göre iç çözünürlük 360p, 480p veya 720p olur.
try:
    window.size = (1920, 1080)
except Exception as _window_size_error:
    print('[KARE ARTTIRICI] 1920x1080 pencere boyutu ayarlanamadı:', _window_size_error)

# Gerçek ekran-uzayı ışın izleme (SSRT): sahne renk + derinlik tamponunda
# ışın yürütme, iki sıçramalı yansıma, derinlik tabanlı temas kararması ve bloom.
# Bu, GPU üzerinde gerçek ışın kesişimi arar; ancak ekran dışındaki/gizli geometriyi
# göremez. Donanımsal DXR/Vulkan RT veya tam dünya-uzayı path tracing değildir.
RT_PIPELINE_AVAILABLE = False
# Spatial upscaling profiles: 360p/480p/720p -> 1080p. Not NVIDIA DLSS/AI frame generation.
UPSCALE_PROFILES = {
    # Profile: (internal render width, internal render height, output width, output height)
    'Performans': (640, 360, 1920, 1080),
    'Dengeli': (854, 480, 1920, 1080),
    'Yüksek': (1280, 720, 1920, 1080),
}
upscaler_profile = 'Yüksek'
UPSCALE_SOURCE_WIDTH, UPSCALE_SOURCE_HEIGHT, UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT = UPSCALE_PROFILES[upscaler_profile]
upscaler_enabled = True
rt_manager = None
rt_quad = None
rt_color_tex = None
rt_depth_tex = None
rt_reflection_strength = 0.0
rt_ao_strength = 0.0
rt_bloom_strength = 0.0
rt_trace_steps = 0
rt_bounces = 0

RT_VERTEX_SHADER = r'''#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
'''

RT_FRAGMENT_SHADER = r'''#version 130
uniform sampler2D sceneColor;
uniform sampler2D sceneDepth;
uniform mat4 projectionMatrix;
uniform mat4 inverseProjectionMatrix;
uniform vec2 resolution;
uniform float reflectionStrength;
uniform float aoStrength;
uniform float bloomStrength;
uniform float framePhase;
uniform int traceSteps;
uniform int maxBounces;
uniform int upscalerEnabled;
uniform float upscaleSharpness;
in vec2 uv;
out vec4 fragColor;

vec3 reconstructView(vec2 p, float d) {
    vec4 clip = vec4(p * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec4 v = inverseProjectionMatrix * clip;
    return v.xyz / max(abs(v.w), 0.000001) * sign(v.w);
}
// Reconstruct positions in the camera/lens space used by Ursina (Y up, +Z forward).
vec3 viewPosition(vec2 p) {
    return reconstructView(p, texture(sceneDepth, clamp(p, vec2(0.001), vec2(0.999))).r);
}
vec3 estimateNormal(vec2 p, vec3 center) {
    vec2 px = vec2(1.0 / max(resolution.x, 1.0), 0.0);
    vec2 py = vec2(0.0, 1.0 / max(resolution.y, 1.0));
    vec2 uvL = clamp(p - px, vec2(0.002), vec2(0.998));
    vec2 uvR = clamp(p + px, vec2(0.002), vec2(0.998));
    vec2 uvD = clamp(p - py, vec2(0.002), vec2(0.998));
    vec2 uvU = clamp(p + py, vec2(0.002), vec2(0.998));
    float depthL = texture(sceneDepth, uvL).r;
    float depthR = texture(sceneDepth, uvR).r;
    float depthD = texture(sceneDepth, uvD).r;
    float depthU = texture(sceneDepth, uvU).r;
    vec3 leftP = viewPosition(uvL);
    vec3 rightP = viewPosition(uvR);
    vec3 downP = viewPosition(uvD);
    vec3 upP = viewPosition(uvU);
    // At silhouettes, pick the smaller depth discontinuity to avoid wildly tilted normals.
    vec3 dx = (depthR < 0.99995 && (depthL >= 0.99995 || abs(rightP.z-center.z) <= abs(center.z-leftP.z)))
              ? (rightP-center) : (center-leftP);
    vec3 dy = (depthU < 0.99995 && (depthD >= 0.99995 || abs(upP.z-center.z) <= abs(center.z-downP.z)))
              ? (upP-center) : (center-downP);
    vec3 n = cross(dx, dy);
    float nLen = length(n);
    if (nLen < 0.00001) n = vec3(0.0, 1.0, 0.0);
    else n /= nLen;
    if (dot(n, -center) < 0.0) n = -n;
    return n;
}
vec2 projectView(vec3 p, out float depthValue) {
    vec4 clip = projectionMatrix * vec4(p, 1.0);
    if (clip.w <= 0.00001) { depthValue = 1.0; return vec2(-2.0); }
    vec3 ndc = clip.xyz / clip.w;
    depthValue = ndc.z * 0.5 + 0.5;
    return ndc.xy * 0.5 + 0.5;
}

// GPU ray-marches invisible reflection rays against the scene depth buffer.
// Returns a depth-tested intersection, refined around front-to-back crossings.
// Ursina configures Panda3D as Y-up / +Z-forward, so view-space Z is linear depth.
bool sampleDepthRay(vec3 origin, vec3 direction, float t,
                    out vec2 suv, out vec3 rayPoint,
                    out vec3 scenePoint, out float delta) {
    rayPoint = origin + direction * t;
    float projectedDepth;
    suv = projectView(rayPoint, projectedDepth);
    if (projectedDepth <= 0.0001 || projectedDepth >= 0.9999 ||
        suv.x <= 0.002 || suv.x >= 0.998 ||
        suv.y <= 0.002 || suv.y >= 0.998) {
        return false;
    }
    float sampledDepth = texture(sceneDepth, suv).r;
    if (sampledDepth >= 0.99995) return false;
    scenePoint = reconstructView(suv, sampledDepth);
    // Negative: ray is in front of the visible surface. Positive: it went behind it (view-space +Z forward).
    delta = rayPoint.z - scenePoint.z;
    return true;
}

bool traceDepthRay(vec3 origin, vec3 direction, out vec2 hitUV, out vec3 hitPos) {
    int steps = clamp(traceSteps, 8, 64);
    // Longer rays for distant origins; quadratic spacing gives more samples close to origin.
    float maxDistance = clamp(length(origin) * 3.2, 12.0, 85.0);
    float previousT = 0.035;
    float previousDelta = -1.0e6;
    bool havePrevious = false;

    for (int i = 1; i <= 64; ++i) {
        if (i > steps) break;
        float u = float(i) / float(steps);
        float t = 0.035 + maxDistance * u * u;
        vec2 suv;
        vec3 rayPoint;
        vec3 scenePoint;
        float delta;
        bool valid = sampleDepthRay(origin, direction, t, suv, rayPoint, scenePoint, delta);
        if (!valid) {
            // A ray may pass through sky or offscreen gaps before hitting another visible object.
            havePrevious = false;
            previousT = t;
            continue;
        }

        float thickness = max(0.055, abs(scenePoint.z) * 0.012);
        bool crossedSurface = havePrevious && previousDelta < 0.0 && delta >= 0.0;
        bool nearSurface = t > 0.30 && delta >= -thickness * 0.35 && delta <= thickness;
        if (crossedSurface || nearSurface) {
            float hitT = t;
            // Refine a front-to-back depth crossing instead of accepting a coarse-step artifact.
            if (crossedSurface) {
                float lo = previousT;
                float hi = t;
                for (int refine = 0; refine < 5; ++refine) {
                    float mid = (lo + hi) * 0.5;
                    vec2 midUV;
                    vec3 midRay;
                    vec3 midScene;
                    float midDelta;
                    if (sampleDepthRay(origin, direction, mid, midUV, midRay, midScene, midDelta)) {
                        if (midDelta < 0.0) lo = mid;
                        else hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                hitT = hi;
                if (!sampleDepthRay(origin, direction, hitT, suv, rayPoint, scenePoint, delta)) {
                    // Outputs are undefined for an invalid sample; reset the crossing history.
                    havePrevious = false;
                    previousT = hitT;
                    continue;
                }
                thickness = max(0.055, abs(scenePoint.z) * 0.012);
            }
            // Reject self-intersections and depth mismatches that are too far apart.
            if (hitT > 0.30 && abs(delta) <= thickness * 2.25) {
                hitUV = suv;
                hitPos = scenePoint;
                return true;
            }
        }
        previousT = t;
        previousDelta = delta;
        havePrevious = true;
    }
    return false;
}

vec3 sampleReflectionColor(vec2 p, float roughness) {
    // Tiny neighborhood filtering reduces stair-stepping on diagonal reflection rays.
    // This is a screen-space roughness approximation, not a per-material PBR roughness map.
    vec2 t = 1.0 / max(resolution, vec2(1.0));
    float r = mix(0.35, 1.6, clamp(roughness, 0.0, 1.0));
    vec3 c = texture(sceneColor, p).rgb * 0.52;
    c += texture(sceneColor, clamp(p + vec2(t.x*r, 0.0), vec2(0.002), vec2(0.998))).rgb * 0.12;
    c += texture(sceneColor, clamp(p - vec2(t.x*r, 0.0), vec2(0.002), vec2(0.998))).rgb * 0.12;
    c += texture(sceneColor, clamp(p + vec2(0.0, t.y*r), vec2(0.002), vec2(0.998))).rgb * 0.12;
    c += texture(sceneColor, clamp(p - vec2(0.0, t.y*r), vec2(0.002), vec2(0.998))).rgb * 0.12;
    return c;
}

vec3 traceScreenPath(vec3 origin, vec3 direction, vec3 initialNormal, out float confidence) {
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    confidence = 0.0;
    vec3 rayOrigin = origin + initialNormal * 0.075;
    vec3 rayDirection = normalize(direction);
    for (int bounce = 0; bounce < 2; ++bounce) {
        if (bounce >= maxBounces) break;
        vec2 hitUV;
        vec3 hitPos;
        if (!traceDepthRay(rayOrigin, rayDirection, hitUV, hitPos)) break;
        float edgeDistance = min(min(hitUV.x, 1.0-hitUV.x), min(hitUV.y, 1.0-hitUV.y));
        float edgeFade = smoothstep(0.015, 0.10, edgeDistance);
        float roughnessApprox = clamp(0.18 + float(bounce) * 0.28, 0.0, 0.8);
        vec3 hitColor = sampleReflectionColor(hitUV, roughnessApprox);
        radiance += hitColor * throughput * edgeFade;
        confidence = max(confidence, edgeFade * (bounce == 0 ? 1.0 : 0.22));
        vec3 hitNormal = estimateNormal(hitUV, hitPos);
        rayDirection = normalize(reflect(rayDirection, hitNormal));
        rayOrigin = hitPos + hitNormal * 0.075;
        throughput *= 0.20;
    }
    return radiance;
}

void main() {
    vec3 baseColor = texture(sceneColor, uv).rgb;
    float depth = texture(sceneDepth, uv).r;
    vec3 result = baseColor;
    vec2 texel = 1.0 / max(resolution, vec2(1.0));

    if (depth < 0.99995) {
        vec3 p = reconstructView(uv, depth);
        vec3 n = estimateNormal(uv, p);

        // Depth-based contact occlusion: actual nearby-depth sampling, not a painted overlay.
        if (aoStrength > 0.001) {
            float occ = 0.0;
            vec2 offsets[8] = vec2[8](
                vec2(1.0,0.0), vec2(-1.0,0.0), vec2(0.0,1.0), vec2(0.0,-1.0),
                vec2(2.0,1.0), vec2(-2.0,1.0), vec2(1.0,-2.0), vec2(-1.0,2.0));
            for (int j=0; j<8; ++j) {
                vec2 q = clamp(uv + offsets[j] * texel * 2.0, vec2(0.001), vec2(0.999));
                float qd = texture(sceneDepth, q).r;
                if (qd < 0.99995) {
                    vec3 qp = reconstructView(q, qd);
                    float nearer = p.z - qp.z;
                    float closeEnough = 1.0 - smoothstep(0.35, 1.7, length(qp-p));
                    occ += smoothstep(0.015, 0.32, nearer) * closeEnough;
                }
            }
            float ao = clamp(1.0 - (occ / 8.0) * aoStrength, 0.42, 1.0);
            result *= ao;
        }

        // Two-bounce screen-space reflection path. It follows traced rays through
        // captured scene depth and shades hits using the real scene-color buffer.
        if (reflectionStrength > 0.001 && uv.y < 0.97) {
            float upFacing = abs(n.y);
            float surfaceMask = smoothstep(0.42, 0.92, upFacing);
            float distanceFade = 1.0 - smoothstep(28.0, 78.0, length(p));
            vec3 viewDirection = normalize(p);
            float facing = clamp(dot(n, -viewDirection), 0.0, 1.0);
            float fresnel = 0.04 + 0.96 * pow(1.0 - facing, 5.0);
            vec3 reflectedDirection = normalize(reflect(viewDirection, n));
            // Keep the ray deterministic per pixel. Temporal jitter without a history buffer
            // causes visible shimmer, so the reflection direction remains stable frame-to-frame.
            float hitConfidence;
            vec3 reflected = traceScreenPath(p, reflectedDirection, n, hitConfidence);
            float fresnelWeight = mix(0.30, 1.0, fresnel);
            float blend = reflectionStrength * surfaceMask * distanceFade * hitConfidence * fresnelWeight;
            // Physically motivated energy blend: preserve the original scene when no ray hits,
            // and never brighten the reflection by adding the base color a second time.
            result = mix(result, reflected, clamp(blend, 0.0, 0.82));
        }
    }

    // HDR-like bloom approximation using bright-pass neighbourhood samples.
    if (bloomStrength > 0.001) {
        vec3 glow = vec3(0.0);
        vec2 b = texel * 2.5;
        vec2 taps[8] = vec2[8](vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1),
                               vec2(1,1),vec2(-1,1),vec2(1,-1),vec2(-1,-1));
        for (int k=0;k<8;++k) {
            vec3 c = texture(sceneColor, clamp(uv + taps[k]*b, vec2(0.001), vec2(0.999))).rgb;
            glow += max(c - vec3(0.68), vec3(0.0));
        }
        result += glow * (bloomStrength / 8.0);
    }
    // Kare Arttırıcı: 720p kaynak örnekleri, tam pencere quad'ı ile 1080p'ye
    // yükseltilir. Bu kenar-duyarlı keskinleştirme, basit bilinear büyütmenin
    // yumuşaklığını azaltır; zamansal/AI DLSS veya kare üretimi değildir.
    if (upscalerEnabled != 0) {
        vec2 st = 1.0 / max(resolution, vec2(1.0));
        vec3 n = texture(sceneColor, clamp(uv + vec2(0.0, st.y), vec2(0.001), vec2(0.999))).rgb;
        vec3 s = texture(sceneColor, clamp(uv - vec2(0.0, st.y), vec2(0.001), vec2(0.999))).rgb;
        vec3 e = texture(sceneColor, clamp(uv + vec2(st.x, 0.0), vec2(0.001), vec2(0.999))).rgb;
        vec3 w = texture(sceneColor, clamp(uv - vec2(st.x, 0.0), vec2(0.001), vec2(0.999))).rgb;
        vec3 lowFrequency = (n + s + e + w) * 0.25;
        vec3 detail = baseColor - lowFrequency;
        float edgeStrength = max(max(abs(detail.r), abs(detail.g)), abs(detail.b));
        float adaptive = mix(0.18, 0.72, smoothstep(0.015, 0.12, edgeStrength));
        result = clamp(result + detail * upscaleSharpness * adaptive, 0.0, 1.5);
    }
    fragColor = vec4(result, 1.0);
}
'''

try:
    from direct.filter.FilterManager import FilterManager
    rt_manager = FilterManager(
        base.win, base.cam,
        forcex=UPSCALE_SOURCE_WIDTH if upscaler_enabled else 0,
        forcey=UPSCALE_SOURCE_HEIGHT if upscaler_enabled else 0
    )
    rt_color_tex = Texture('kk-scene-color')
    rt_depth_tex = Texture('kk-scene-depth')
    rt_quad = rt_manager.renderSceneInto(colortex=rt_color_tex, depthtex=rt_depth_tex)
    rt_shader = Shader.make(Shader.SL_GLSL, RT_VERTEX_SHADER, RT_FRAGMENT_SHADER)
    rt_quad.setShader(rt_shader)
    rt_quad.setShaderInput('sceneColor', rt_color_tex)
    rt_quad.setShaderInput('sceneDepth', rt_depth_tex)
    rt_quad.setShaderInput('upscalerEnabled', 1 if upscaler_enabled else 0)
    rt_quad.setShaderInput('upscaleSharpness', 0.42)
    # Smooth color upsampling; nearest depth avoids mixing unrelated surfaces at edges.
    rt_color_tex.setMinfilter(Texture.FTLinear)
    rt_color_tex.setMagfilter(Texture.FTLinear)
    rt_depth_tex.setMinfilter(Texture.FTNearest)
    rt_depth_tex.setMagfilter(Texture.FTNearest)
    RT_PIPELINE_AVAILABLE = True
    print('[SSRT] GPU ekran-uzayı ışın izleme hattı etkin.')
except Exception as rt_error:
    RT_PIPELINE_AVAILABLE = False
    rt_manager = None
    rt_quad = None
    print('[SSRT] GPU ışın izleme açılamadı; temel render kullanılacak:', rt_error)


def update_raytrace_uniforms():
    if not RT_PIPELINE_AVAILABLE or rt_quad is None:
        return
    try:
        proj = Mat4(base.cam.node().getLens().getProjectionMat())
        inv_proj = Mat4(proj)
        inv_proj.invertInPlace()
        rt_quad.setShaderInput('projectionMatrix', proj)
        rt_quad.setShaderInput('inverseProjectionMatrix', inv_proj)
        if upscaler_enabled:
            src_w, src_h = UPSCALE_SOURCE_WIDTH, UPSCALE_SOURCE_HEIGHT
        else:
            src_w, src_h = int(window.size[0]), int(window.size[1])
        rt_quad.setShaderInput('resolution', Vec2(max(1, src_w), max(1, src_h)))
        rt_quad.setShaderInput('framePhase', float(_time.monotonic()))
    except Exception as matrix_error:
        # Do not stop gameplay if a driver rejects a shader uniform update.
        pass


def set_postfx_quality(level):
    global rt_reflection_strength, rt_ao_strength, rt_bloom_strength, rt_trace_steps, rt_bounces
    enabled = bool(globals().get('advanced_lighting_enabled', True))
    if level == 'Düşük':
        rt_reflection_strength = 0.0
        rt_ao_strength = 0.12 if enabled else 0.0
        rt_bloom_strength = 0.08 if enabled else 0.0
        rt_trace_steps, rt_bounces = 12, 1
    elif level == 'Orta':
        rt_reflection_strength = 0.36 if enabled else 0.0
        rt_ao_strength = 0.30 if enabled else 0.0
        rt_bloom_strength = 0.18 if enabled else 0.0
        rt_trace_steps, rt_bounces = 24, 1
    else:
        rt_reflection_strength = 0.62 if enabled else 0.0
        rt_ao_strength = 0.48 if enabled else 0.0
        rt_bloom_strength = 0.42 if enabled else 0.0
        rt_trace_steps, rt_bounces = 48, 2
    if RT_PIPELINE_AVAILABLE and rt_quad is not None:
        try:
            rt_quad.setShaderInput('reflectionStrength', float(rt_reflection_strength))
            rt_quad.setShaderInput('aoStrength', float(rt_ao_strength))
            rt_quad.setShaderInput('bloomStrength', float(rt_bloom_strength))
            rt_quad.setShaderInput('traceSteps', int(rt_trace_steps))
            rt_quad.setShaderInput('maxBounces', int(rt_bounces))
        except Exception as shader_error:
            print('[SSRT] Kalite ayarı uygulanamadı:', shader_error)
    update_raytrace_uniforms()

def cyl():
    return Cylinder(8, start=-.5)

window.title      = 'Karanlık Kral – 3D Macera'
window.borderless = False
window.fullscreen = False
window.fps_counter.enabled = True
window.exit_button.visible = False
window.color = color.black

# ── SICAK YEREL IŞIKLAR: MEŞALE + LAV ─────────────────────────
# Ursina'nın standart lit_with_shadows_shader'ı yalnızca tek yönlü ışık kaynağını
# gölgeyle işler. Bu türev aynı gölge hesabını korurken ayrıca görünmez yerel
# ışık kaynaklarının her yüzeyde mesafeye ve yüzey normaline göre katkısını hesaplar.
MAX_WARM_LIGHTS = 16
LOCAL_WARM_LIGHTS = []

_warm_uniforms = '\n'.join(
    f'uniform vec4 kkWarmPos{i};\nuniform vec4 kkWarmColor{i};'
    for i in range(MAX_WARM_LIGHTS)
)
_warm_code_parts = []
for _i in range(MAX_WARM_LIGHTS):
    _warm_code_parts.append(f'''    {{
        vec3 toWarm{_i} = kkWarmPos{_i}.xyz - vertex_world_position;
        float distWarm{_i} = length(toWarm{_i});
        float rangeWarm{_i} = max(kkWarmPos{_i}.w, 0.001);
        float fadeWarm{_i} = clamp(1.0 - distWarm{_i} / rangeWarm{_i}, 0.0, 1.0);
        float attenWarm{_i} = fadeWarm{_i} * fadeWarm{_i};
        vec3 dirWarm{_i} = toWarm{_i} / max(distWarm{_i}, 0.001);
        float ndlWarm{_i} = max(dot(normalize(warm_world_normal), dirWarm{_i}), 0.0);
        float wrapWarm{_i} = clamp((dot(normalize(warm_world_normal), dirWarm{_i}) + 0.35) / 1.35, 0.0, 1.0);
        localWarm += kkWarmColor{_i}.rgb * kkWarmColor{_i}.a * attenWarm{_i} * (0.22 + 0.78 * wrapWarm{_i});
    }}''')
_warm_code = '\n'.join(_warm_code_parts)
try:
    _base_vertex = lit_with_shadows_shader.vertex
    # Local lights are expressed in world space, so use a world-space normal for them.
    _base_vertex = _base_vertex.replace(
        'out vec3 vertex_world_position;',
        'out vec3 vertex_world_position;\nout vec3 warm_world_normal;'
    ).replace(
        'normal_vector = normalize(p3d_NormalMatrix * normal);',
        'normal_vector = normalize(p3d_NormalMatrix * normal);\n'
        '    warm_world_normal = normalize(transpose(inverse(mat3(p3d_ModelMatrix))) * normal);'
    )
    _base_fragment = lit_with_shadows_shader.fragment
    _new_fragment = _base_fragment.replace(
        'in vec3 normal_vector;',
        'in vec3 normal_vector;\nin vec3 warm_world_normal;\n' + _warm_uniforms
    )
    _warm_injection = '''fragment_color = cast_shadows(fragment_color);
    vec3 localWarm = vec3(0.0);
''' + _warm_code + '''
    vec3 warmAlbedo = texture(p3d_Texture0, texcoords).rgb * p3d_ColorScale.rgb * vertex_color.rgb;
    fragment_color.rgb += warmAlbedo * min(localWarm, vec3(1.8));'''
    if _new_fragment == _base_fragment or 'fragment_color = cast_shadows(fragment_color);' not in _new_fragment:
        raise RuntimeError('Beklenen gölge shader kodu bulunamadı')
    _new_fragment = _new_fragment.replace('fragment_color = cast_shadows(fragment_color);', _warm_injection, 1)
    torch_lighting_shader = UrsinaShader(
        name='kk_torch_aware_shadows_shader',
        language=UrsinaShader.GLSL,
        vertex=_base_vertex,
        fragment=_new_fragment
    )
    # Compile immediately so incompatible GPU drivers fall back before gameplay starts.
    torch_lighting_shader.compile(shader_includes=False)
    torch_lighting_shader.default_input = dict(lit_with_shadows_shader.default_input)
    torch_lighting_shader.continuous_input = dict(lit_with_shadows_shader.continuous_input)
    TORCH_SHADER_AVAILABLE = True
    print('[ATEŞ IŞIĞI] Yerel meşale/lav aydınlatma shaderı hazır.')
except Exception as warm_shader_error:
    torch_lighting_shader = lit_with_shadows_shader
    TORCH_SHADER_AVAILABLE = False
    print('[ATEŞ IŞIĞI] Özel shader kullanılamadı, standart gölge shaderı kullanılacak:', warm_shader_error)

for _i in range(MAX_WARM_LIGHTS):
    render.setShaderInput(f'kkWarmPos{_i}', Vec4(0, -1000, 0, 0.01))
    render.setShaderInput(f'kkWarmColor{_i}', Vec4(0, 0, 0, 0))

def register_warm_light(node, tint=(1.0, 0.38, 0.08), radius=9.0, intensity=1.4):
    # Görünmez, GPU shader'ında hesaplanan yerel sıcak ışık kaynağı.
    source = {
        'node': node,
        'tint': tuple(max(0.0, min(1.0, float(c))) for c in tint),
        'radius': float(radius),
        'intensity': float(intensity),
        'phase': random.uniform(0.0, math.tau),
    }
    LOCAL_WARM_LIGHTS.append(source)
    return source

def register_warm_light_at(owner, position, tint=(1.0, 0.30, 0.04), radius=10.0, intensity=1.5):
    # Görünmez çıpa meşalenin/lavın koordinatını shader'a taşır; ışının kendisi görünmez.
    anchor = Entity(model='sphere', position=position, scale=0.001,
                    visible=False, collider=None, shader=None)
    if hasattr(owner, 'ents'):
        owner.ents.append(anchor)
    return register_warm_light(anchor, tint=tint, radius=radius, intensity=intensity)

def update_warm_light_uniforms():
    if not TORCH_SHADER_AVAILABLE:
        return
    alive = []
    for source in list(LOCAL_WARM_LIGHTS):
        node = source.get('node')
        try:
            if node is None or node.is_empty():
                LOCAL_WARM_LIGHTS.remove(source)
                continue
            pos = node.world_position
            d2 = (pos.x - player.x) ** 2 + (pos.y - player.y) ** 2 + (pos.z - player.z) ** 2 if 'player' in globals() else 0.0
            alive.append((d2, source, pos))
        except Exception:
            try: LOCAL_WARM_LIGHTS.remove(source)
            except ValueError: pass
    # Select nearby lamps first so the fixed GPU light budget follows the active area.
    alive.sort(key=lambda item: item[0])
    now = _time.monotonic()
    for i in range(MAX_WARM_LIGHTS):
        if i < len(alive):
            _d2, source, pos = alive[i]
            flicker = 0.90 + 0.10 * (0.5 + 0.5 * math.sin(now * 13.0 + source['phase']))
            tint = source['tint']
            render.setShaderInput(f'kkWarmPos{i}', Vec4(pos.x, pos.y, pos.z, source['radius']))
            render.setShaderInput(f'kkWarmColor{i}', Vec4(tint[0], tint[1], tint[2], source['intensity'] * flicker))
        else:
            render.setShaderInput(f'kkWarmPos{i}', Vec4(0, -1000, 0, 0.01))
            render.setShaderInput(f'kkWarmColor{i}', Vec4(0, 0, 0, 0))

def add_warm_ground_glow(owner, x, z, floor_y=0.015, radius=2.2, tint=(255, 105, 24)):
    # Katmanlı zemin parıltısı; gerçek yüzey aydınlatmasını shader yapar.
    glows = []
    # Outer layer is deep orange, the middle is amber, the core tends towards yellow.
    r, green, blue = tint
    layers = [
        (1.00, 10, (r, max(18, int(green * 0.55)), max(2, int(blue * 0.7)))),
        (0.70, 13, (r, green, blue)),
        (0.42, 18, (r, min(255, int(green * 1.45 + 35)), min(255, int(blue * 1.25 + 12)))),
    ]
    for scale_factor, alpha, layer_tint in layers:
        g = owner._mk(model='sphere', position=(x, floor_y, z),
                      scale=(radius * scale_factor, 0.018, radius * scale_factor),
                      color=color.rgba32(layer_tint[0], layer_tint[1], layer_tint[2], alpha),
                      unlit=True, shader=None, transparent=True, collider=None)
        glows.append(g)
    return glows

# ── IŞIK, GERÇEK GÖLGELER VE GERÇEK ZAMANLI YANSIMA ───────────
night_ambient = AmbientLight(color=color.rgba32(15, 18, 34, 255))

# DirectionalLight shadow map üretir. lit_with_shadows_shader kullanan
# modeller bu haritadan gerçek geometrik gölge alır.
sun = DirectionalLight(shadows=True, shadow_map_resolution=Vec2(2048, 2048))
sun.color = color.rgb32(105, 122, 190)  # soğuk ay ışığı
sun.look_at(Vec3(-0.8, -2.0, -0.5))
# Gölge kamerasının sınırları yalnızca boş sahneye göre hesaplanırsa gölgeler kaybolur.
# Bu görünmez hacim üç bölümün tamamını kapsayan kararlı bir gölge alanı tanımlar.
shadow_bounds = Entity(model='cube', position=(0, 6, 8), scale=(42, 20, 78),
                       color=color.rgba32(0, 0, 0, 0), visible=False, collider=None)
try:
    sun._light.setShadowCaster(True, 2048, 2048)
    sun.update_bounds(shadow_bounds)
except Exception as shadow_error:
    print('[GÖLGE] Başlangıçta gölge alanı kurulamadı:', shadow_error)


def mark_glow(entity):
    # Panda3D sürümleri arasında olmayan GlowAttrib'a bağımlı değiliz.
    # Parlak malzemeler birleşik post-process bloom filtresinde örneklenir.
    return entity

# Planar reflection: ikinci kamera sahneyi zemin düzlemine göre aynalayarak
# ayrı bir texture'a çizer. Bu, sahte parlaklık değil, sahnenin gerçek
# zamanlı render edilmiş yansımasıdır. Yansıma zemin yüzeylerine uygulanır.
REFLECTION_PLANE_Y = -0.1
REFLECTION_MASK = BitMask32.bit(1)
# The main camera must NOT include the dedicated reflection mask. Otherwise Panda3D's
# hide(REFLECTION_MASK) would hide the floor from the player camera as well.
try:
    base.cam.node().setCameraMask(BitMask32.allOn() & ~REFLECTION_MASK)
except Exception as camera_mask_error:
    print('[YANSIMA] Kamera maskesi ayarlanamadı:', camera_mask_error)
REFLECTION_STAGE = TextureStage('real-time-ground-reflection')
REFLECTION_STAGE.setMode(TextureStage.MDecal)
REFLECTION_STAGE.setSort(1)
REFLECTION_SURFACES = []
reflection_enabled = False
reflection_camera = None
reflection_buffer = None
reflection_texture = None
try:
    reflection_buffer = base.win.makeTextureBuffer('KaranlikKral_Reflection', 1024, 1024)
    if reflection_buffer:
        reflection_buffer.setSort(-100)
        try:
            reflection_buffer.setClearColor(Vec4(0, 0, 0, 0))
            reflection_buffer.setClearColorActive(True)
        except Exception:
            pass
        reflection_texture = reflection_buffer.getTexture()
        reflection_camera = base.makeCamera(reflection_buffer)
        reflection_camera.node().setCameraMask(REFLECTION_MASK)
        try:
            reflection_camera.node().getLens().setNearFar(0.1, 500)
            reflection_camera.node().getLens().setFov(camera.lens.getFov())
            reflection_buffer.setActive(False)  # kalite seçeneği etkinleştirince açılır
        except Exception:
            pass
        reflection_enabled = reflection_texture is not None
        if RT_PIPELINE_AVAILABLE:
            reflection_buffer.setActive(False)
except Exception as reflection_error:
    print('[YANSIMA] Gerçek zamanlı yansıma başlatılamadı:', reflection_error)


def _set_surface_reflection(surface, enabled):
    try:
        if RT_PIPELINE_AVAILABLE:
            try:
                surface.clearProjectTexture(REFLECTION_STAGE)
            except Exception:
                pass
            surface.shader = torch_lighting_shader
            return
        if enabled and reflection_buffer is not None and reflection_camera is not None:
            # Keep each reflective floor tile out of the reflection camera to avoid feedback.
            surface.hide(REFLECTION_MASK)
            surface.shader = None
            surface.setShaderAuto()
            surface.projectTexture(REFLECTION_STAGE, reflection_texture, reflection_camera)
        else:
            try:
                surface.clearProjectTexture(REFLECTION_STAGE)
            except Exception:
                pass
            surface.shader = torch_lighting_shader
    except Exception as reflection_error:
        print('[YANSIMA] Yüzey güncellenemedi:', reflection_error)


def set_reflection_pass(enabled):
    # Toggle ray-marched reflections; only use camera-projected fallback on unsupported builds.
    enabled = bool(enabled and (RT_PIPELINE_AVAILABLE or reflection_enabled))
    if RT_PIPELINE_AVAILABLE:
        try:
            rt_quad.setShaderInput('reflectionStrength', float(rt_reflection_strength if enabled else 0.0))
        except Exception:
            pass
    if reflection_buffer is not None:
        try:
            # SSR zaten sahneyi derinlik tamponundan izliyorsa ikinci kamera buffer'ı gereksiz.
            reflection_buffer.setActive(bool(enabled and not RT_PIPELINE_AVAILABLE))
        except Exception:
            pass
    for surface in list(REFLECTION_SURFACES):
        if surface is not None and not surface.is_empty():
            _set_surface_reflection(surface, enabled)


def register_reflection_surface(surface):
    """Register a floor/lava mesh for a live secondary-camera reflection pass."""
    if not reflection_enabled:
        return
    REFLECTION_SURFACES.append(surface)
    # This is a real secondary-camera scene render, not a painted fake highlight.
    _set_surface_reflection(surface, advanced_lighting_enabled)

def update_reflection_camera():
    if not reflection_enabled or reflection_camera is None:
        return
    try:
        # Kamerayı y=REFLECTION_PLANE_Y düzleminin altına aynala.
        reflection_camera.setPos(
            camera.getX(render),
            2 * REFLECTION_PLANE_Y - camera.getY(render),
            camera.getZ(render)
        )
        reflection_camera.setHpr(
            camera.getH(render), -camera.getP(render), -camera.getR(render)
        )
    except Exception:
        pass

# ── OYUN DURUMU ───────────────────────────────────────────────
class G:
    chapter    = 1
    hp         = 100
    has_sword  = False
    has_key    = False
    dlg_open   = False
    enemy_dead = False
    atk_cd     = 0.0
    respawn    = False
g = G()

# ─────────────────────────────────────────────────────────────
#  DİYALOG
# ─────────────────────────────────────────────────────────────
import textwrap

def wrap_text(msg, width=58):
    out = []
    for para in msg.split('\n'):
        out.append('\n'.join(textwrap.wrap(para, width)) if para.strip() else '')
    return '\n'.join(out)

class Dlg(Entity):
    PW   = 1.16     # panel genişliği
    BOT  = -0.47    # panel alt kenarı
    TS   = 1.2      # yazı ölçeği
    LH   = 1.5      # satır aralığı
    LINE = 0.025 * 1.2 * 1.5   # bir satırın yüksekliği

    def __init__(self):
        super().__init__(parent=camera.ui, enabled=False, z=-10)
        self.panel  = Entity(parent=self, model='quad', z=0,
                             color=color.rgba32(8,5,18,240))
        self.header = Entity(parent=self, model='quad', z=-0.01,
                             color=color.rgba32(60,25,110,255))
        self.ttl = Text('', parent=self, scale=1.35, origin=(-.5,.5), z=-0.02,
                        color=color.rgb32(220,195,255))
        self.bod = Text('', parent=self, scale=self.TS, origin=(-.5,.5), z=-0.02,
                        line_height=self.LH, color=color.rgb32(240,240,240))
        self.hnt = Text('[ ENTER ]  Altyazıyı Atla', parent=self, scale=1.0,
                        origin=(.5,-.5), z=-0.02,
                        color=color.rgb32(140,200,255))
        self.saved = None
        self._show_id = 0

    def show(self, msg, title='Hikaye'):
        body = wrap_text(msg)
        n    = body.count('\n') + 1
        hdr  = 0.065
        h    = hdr + 0.035 + n * self.LINE + 0.055
        top  = self.BOT + h
        left = -self.PW/2 + 0.03

        self.panel.scale  = (self.PW, h)
        self.panel.position = (0, self.BOT + h/2)
        self.header.scale = (self.PW, hdr)
        self.header.position = (0, top - hdr/2)
        self.ttl.text = title
        self.ttl.position = (left, top - 0.016)
        self.bod.text = body
        self.bod.position = (left, top - hdr - 0.02)
        self.hnt.position = (self.PW/2 - 0.025, self.BOT + 0.012)

        self.enabled = True
        g.dlg_open   = True
        # Altyazı oyunu durdurmaz: oyuncu yürüyebilir, zıplayabilir ve oynanış devam eder.
        # Kamera kontrolü oyun sırasında kilitli kalır; fare menü imlecine dönüşmez.
        mouse.locked = not settings_open
        mouse.visible = settings_open

        # Yeni altyazı kendi 7 saniyelik süresini başlatır.
        # Önceki zamanlayıcı yeni altyazıyı kapatamaz.
        self._show_id += 1
        show_id = self._show_id
        invoke(lambda: self._auto_hide(show_id), delay=7)

    def _auto_hide(self, show_id):
        if self.enabled and show_id == self._show_id:
            self.hide()

    def hide(self):
        self.enabled = False
        g.dlg_open   = False
        self._show_id += 1
        # Diyaloğun oyuncu hareket hızına dokunmaması gerekir.
        self.saved = None
        mouse.locked = not settings_open
        mouse.visible = settings_open
        if g.respawn:
            g.respawn = False
            g.hp = 100
            g.has_key = False
            load_chapter(g.chapter)

dlg = Dlg()

# ─────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────
class HUD(Entity):
    def __init__(self):
        super().__init__(parent=camera.ui)
        # HP arka + bar
        Entity(parent=self, model='quad', scale=(0.21,0.028),
               position=(-0.755,0.463), color=color.rgba32(0,0,0,170))
        self.hp_bar = Entity(parent=self, model='quad',
                             scale=(0.21,0.021), position=(-0.755,0.463),
                             color=color.rgb32(200,50,50))
        self.hp_txt = Text('HP: 100', parent=self,
                           position=(-0.845,0.452), scale=1.35, color=color.white)
        # Bölüm
        self.ch_txt = Text('', parent=self,
                           position=(-0.845,0.426), scale=1.2,
                           color=color.rgb32(160,210,255))
        # Envanter kutusu
        Entity(parent=self, model='quad', scale=(0.21,0.075),
               position=(-0.755,0.374), color=color.rgba32(0,0,0,150))
        self.inv = Text('', parent=self,
                        position=(-0.845,0.390), scale=1.15,
                        color=color.rgb32(255,215,80), line_height=1.6)
        # Şato bulmacasında bulunan sayı kâğıtları ekranda köşede tutulur.
        self.code_note_bg = Entity(parent=self, model='quad', scale=(0.37,0.19),
                                   position=(0.655,0.355), color=color.rgba32(5,8,16,205),
                                   enabled=False)
        self.code_note = Text('', parent=self, position=(0.49,0.435),
                              scale=0.82, origin=(-0.5,0.5),
                              color=color.rgb32(245,220,150), enabled=False)
        # Ortadaki ipucu
        self.hint = Text('', parent=self, origin=(0,0),
                         position=(0,-0.42), scale=1.55,
                         color=color.rgb32(255,255,80))
        # Boss can barı (ekranın üstünde)
        self.boss_bg   = Entity(parent=self, model='quad', scale=(0.64,0.036),
                                position=(0,0.43), color=color.rgba32(0,0,0,200),
                                enabled=False)
        self.boss_fill = Entity(parent=self, model='quad', scale=(0.62,0.026),
                                position=(0,0.43), z=-0.01,
                                color=color.rgb32(200,30,30), enabled=False)
        self.boss_name = Text('KARANLIK ŞÖVALYE', parent=self, origin=(0,0),
                              position=(0,0.465), scale=1.2,
                              color=color.rgb32(255,120,120), enabled=False)
        # Duyuru, vuruş sayısı, kırmızı hasar ekranı
        self.msg   = Text('', parent=self, origin=(0,0), position=(0,0.33),
                          scale=1.8, color=color.rgb32(255,170,60))
        self.pop   = Text('', parent=self, origin=(0,0), position=(0,0.07),
                          scale=2.2, color=color.rgb32(255,240,120))
        self.flash = Entity(parent=self, model='quad', scale=(3,2), z=1,
                            color=color.rgba32(255,0,0,0))
        self._msg_id = 0
        self._pop_id = 0
        self.refresh()

    def set_boss(self, ratio):
        on = ratio is not None
        for e in (self.boss_bg, self.boss_fill, self.boss_name):
            e.enabled = on
        if on:
            r = max(0.0, min(1.0, ratio))
            self.boss_fill.scale_x = max(0.0001, 0.62 * r)
            self.boss_fill.x = -0.31 + 0.31 * r

    def announce(self, text, secs=2.2):
        self.msg.text = text
        self._msg_id += 1
        mid = self._msg_id
        invoke(lambda: self._clear_msg(mid), delay=secs)

    def _clear_msg(self, mid):
        if self._msg_id == mid:
            self.msg.text = ''

    def popup(self, text):
        self.pop.text = text
        self.pop.y = 0.07
        self.pop.animate('y', 0.14, duration=0.35)
        self._pop_id += 1
        pid = self._pop_id
        invoke(lambda: self._clear_pop(pid), delay=0.4)

    def _clear_pop(self, pid):
        if self._pop_id == pid:
            self.pop.text = ''

    def hurt(self):
        self.flash.color = color.rgba32(255,0,0,120)
        self.flash.animate_color(color.rgba32(255,0,0,0), duration=0.4)

    def refresh(self):
        self.hp_txt.text = f'HP: {g.hp}'
        r = 0.21 * (g.hp/100)
        self.hp_bar.scale_x = max(r,0)
        self.hp_bar.x = -0.755 - (0.21-r)/2
        ri = int(200*(1-g.hp/100)); gi = int(200*(g.hp/100))
        self.hp_bar.color = color.rgb32(ri,gi,20)
        self.ch_txt.text = f'Bölüm {g.chapter} / 4'
        inv = []
        if g.has_sword: inv.append('Kılıç')
        if g.has_key:   inv.append('Anahtar')
        self.inv.text = '\n'.join(inv) if inv else 'Boş'

    def set_hint(self,t): self.hint.text = t

hud = HUD()

# ── GECE GÖKYÜZÜ / AY / YILDIZLAR ─────────────────────────────
def add_night_sky(owner, include_moon=True, star_count=95):
    """Sahne küresinin içine yerleştirilmiş, sabit ve çarpışmasız gece gökyüzü dekoru."""
    # Yıldızlar üst yarımküreye dağıtılır; oyuncuya çarpışma veya gölge yükü eklemezler.
    for _ in range(star_count):
        angle = random.uniform(0.0, math.tau)
        elevation = random.uniform(0.10, 1.0)
        horiz = math.sqrt(max(0.0, 1.0 - elevation * elevation))
        radius = random.uniform(218.0, 232.0)
        pos = (math.cos(angle) * horiz * radius,
               elevation * radius,
               math.sin(angle) * horiz * radius)
        size = random.uniform(0.18, 0.48)
        star_color = random.choice([
            color.rgb32(165, 190, 255),
            color.rgb32(205, 220, 255),
            color.rgb32(255, 227, 190),
        ])
        star = owner._mk(model='sphere', position=pos, scale=size,
                         color=star_color, unlit=True, shader=None,
                         collider=None)

    if include_moon:
        moon_pos = Vec3(-42, 205, 90)
        # Transparan dış halo, ardından ay diski. Her ikisi de ışık haritasına girmez.
        owner._mk(model='sphere', position=moon_pos, scale=22,
                  color=color.rgba32(85, 112, 190, 28), unlit=True,
                  shader=None, transparent=True, collider=None)
        moon = owner._mk(model='sphere', position=moon_pos, scale=11.5,
                         color=color.rgb32(190, 204, 235), unlit=True,
                         shader=None, collider=None)
        # Küçük soğuk ton farkları ay yüzeyine krater hissi verir.
        for dx, dy, dz, size in [(-1.4, 1.5, -5.2, 1.0),
                                 (1.7, -0.8, -5.0, 0.72),
                                 (0.3, 2.4, -4.8, 0.48)]:
            owner._mk(model='sphere', position=moon_pos + Vec3(dx, dy, dz),
                      scale=size, color=color.rgb32(155, 173, 211),
                      unlit=True, shader=None, collider=None)


def add_low_mist(owner, positions, tint=(45, 62, 105)):
    """Yere yapışık, çok hafif hacim hissi veren sis şeritleri."""
    for x, z, width in positions:
        owner._mk(model='sphere', position=(x, 0.20, z),
                  scale=(width, 0.12, random.uniform(1.6, 3.0)),
                  color=color.rgba32(tint[0], tint[1], tint[2], 22),
                  unlit=True, shader=None, transparent=True, collider=None)


# ─────────────────────────────────────────────────────────────
#  BÖLÜM 1 – ORMAN
# ─────────────────────────────────────────────────────────────
class Chapter1(Entity):
    def __init__(self):
        super().__init__()
        self.ents = []
        self.torch_flames = []
        self.elapsed = 0.0
        self.atmosphere_fog_color = color.rgb32(7, 12, 27)
        self.atmosphere_fog_density = 0.026
        self._build()
        invoke(lambda: dlg.show(
            "Kayıp Prenses'i bulmak için\n"
            "bu ormana girdin.\n\n"
            "Parlayan bir şey dikkatini çekti...\n"
            "Kaleye giden kapıyı bul.",
            'Bölüm 1 – Karanlık Orman'), delay=0.6)

    def _mk(self,*a,**kw):
        # Dünya modellerinde gölge alabilen ışıklandırma shader'ı.
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e=Entity(*a,**kw); self.ents.append(e)
        try:
            sc=e.scale
            dims=(abs(float(sc.x)),abs(float(sc.y)),abs(float(sc.z)))
            # Uzakta pahalı olmayan, küçük dekoratif ayrıntılar LOD ile render dışına alınabilir.
            # Collider taşıyan duvar/zemin/mobilya ve çok büyük gök kubbeleri korunur.
            e._kk_lod_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 3.6 and max(dims) > 0.015)
            e._kk_occlusion_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 12.0 and max(dims) > 0.015)
        except Exception:
            e._kk_lod_candidate=False
            e._kk_occlusion_candidate=False
        return e

    def _make_torch(self, x, z, base_y=0.0, pole_h=1.15):
        pole = self._mk(model=cyl(), scale=(0.09, pole_h, 0.09),
                        position=(x, base_y + pole_h / 2, z),
                        color=color.rgb32(65, 39, 20))
        # Kısa yatay tutucu, ateşin direğin ucuna bağlı görünmesini sağlar.
        self._mk(model='cube', scale=(0.24, 0.06, 0.08),
                 position=(x, base_y + pole_h - 0.05, z),
                 color=color.rgb32(75, 46, 22))
        flame_y = base_y + pole_h + 0.08
        flame = self._mk(model='sphere', scale=0.28,
                         position=(x, flame_y, z),
                         color=color.rgb32(255, 118, 24),
                         unlit=True, shader=None)
        inner = self._mk(model='sphere', scale=0.14,
                         position=(x, flame_y + 0.06, z),
                         color=color.rgb32(255, 205, 95),
                         unlit=True, shader=None)
        mark_glow(flame)
        light = register_warm_light(flame, tint=(1.0, 0.39, 0.10), radius=11.5, intensity=1.75)
        add_warm_ground_glow(self, x, z, floor_y=base_y + 0.012, radius=2.0,
                             tint=(255, 105, 26))
        self.torch_flames.append((flame, inner, light, flame_y, random.uniform(0, math.tau)))

    def _build(self):
        # Gece göğü: ay, yıldızlar ve derin lacivert kubbe.
        self._mk(model='sphere', scale=500,
                 color=color.rgb32(3, 5, 17), double_sided=True, unlit=True, shader=None)
        add_night_sky(self, include_moon=True, star_count=110)
        self.atmosphere_fog_color = color.rgb32(7, 12, 27)
        self.atmosphere_fog_density = 0.026
        scene.fog_color = self.atmosphere_fog_color
        scene.fog_density = self.atmosphere_fog_density

        # Zemin – koyu yeşil, mozaik kareler
        for gx in range(-8,9):
            for gz in range(-8,9):
                c_ = color.rgb32(
                    random.randint(18,30),
                    random.randint(50,80),
                    random.randint(18,30))
                floor_tile = self._mk(model='cube', collider='box',
                         position=(gx*2, -0.1, gz*2),
                         scale=(2.02, 0.2, 2.02), color=c_, shader=None if reflection_enabled else torch_lighting_shader)
                register_reflection_surface(floor_tile)

        # BÖLÜM 1 SINIRLARI: oyuncunun harita dışına yürüyerek çıkmasını önleyen
        # koyu taş/kök karışımı çevre duvarları. Kuzey kapısı duvarın içinde kalır.
        boundary_color = color.rgb32(27, 36, 28)
        for wall_pos, wall_scale in [
            ((-17.1, 3.8, 0), (0.65, 7.6, 34.2)),
            (( 17.1, 3.8, 0), (0.65, 7.6, 34.2)),
            ((0, 3.8, -17.1), (34.2, 7.6, 0.65)),
            ((0, 3.8,  17.1), (34.2, 7.6, 0.65)),
        ]:
            self._mk(model='cube', position=wall_pos, scale=wall_scale,
                     color=boundary_color, collider='box')

        # Alçak, seyrek orman sisi. Sis yere yakın tutulur.
        add_low_mist(self, [(-8, -1, 5.5), (6, 2, 6.0), (-3, 8, 4.5),
                            (9, -7, 5.0), (-10, 10, 4.0)], (30, 52, 88))

        # === AĞAÇLAR ===
        tree_pos = [
            (-5,-4),(-8,0),(-4,5),(-11,2),(-6,9),(-9,-8),
            (5,-3),(9,1),(6,7),(12,-1),(4,11),(10,-7),
            (-2,-10),(2,-11),(0,13),(-13,-3),(13,5),
            (-7,13),(7,-11),(14,9),(-14,7)
        ]
        for tx,tz in tree_pos:
            h  = random.uniform(4.0,7.0)
            r  = random.uniform(0.22,0.38)
            br = random.randint(45,70)
            bg = random.randint(28,45)
            bb = random.randint(15,28)
            # Gövde (koyu kahve)
            self._mk(model=cyl(), scale=(r*2,h,r*2),
                     position=(tx, h/2, tz),
                     color=color.rgb32(br,bg,bb))
            # Taç katmanları (3 koyu yeşil küre)
            for ly,ls,da in [(h*0.60,3.2,0),(h*0.75,2.5,8),(h*0.90,1.6,15)]:
                gc = random.randint(30+da,55+da)
                self._mk(model='sphere', scale=ls,
                         position=(tx,ly,tz),
                         color=color.rgb32(15,gc,15))

        # Kayalar
        for _ in range(14):
            px=random.uniform(-13,13); pz=random.uniform(-13,13)
            s =random.uniform(0.35,0.95)
            self._mk(model='cube', collider='box',
                     position=(px,s*0.35,pz),
                     scale=(s*1.4,s*0.65,s*1.1),
                     color=color.rgb32(
                         random.randint(70,100),
                         random.randint(70,100),
                         random.randint(75,110)))

        # Küçük çalılar
        for _ in range(22):
            px=random.uniform(-13,13); pz=random.uniform(-13,13)
            self._mk(model='sphere',
                     position=(px,0.28,pz),
                     scale=random.uniform(0.35,0.75),
                     color=color.rgb32(
                         random.randint(15,35),
                         random.randint(60,100),
                         random.randint(15,35)))

        # Meşaleler: direk tabanları zemine tam oturur, alevler hafifçe titreşir.
        for tx,tz in [(-1.5,4),(-1.5,8),(1.5,4),(1.5,8)]:
            self._make_torch(tx, tz, base_y=0.0, pole_h=1.15)

        # === KILIÇ ===
        self.sword = self._mk(
            model='cube', scale=(0.1,0.85,0.1),
            position=(4,0.5,-3), collider='box',
            color=color.rgb32(180,190,210))
        # Kılıç koruyucu (yatay çubuk)
        self._mk(model='cube', scale=(0.55,0.08,0.08),
                 position=(4,0.65,-3),
                 color=color.rgb32(200,160,60))
        # Parıltı
        self._mk(model='sphere',scale=0.55,
                 position=(4,0.8,-3),
                 color=color.rgba32(150,180,255,55))

        # === KAPI (kuzey) ===
        for sx in [-1.4,1.4]:
            self._mk(model='cube',scale=(0.28,4.5,0.28),
                     position=(sx,2.25,12),
                     color=color.rgb32(65,40,18))
        self._mk(model='cube',scale=(3.1,0.3,0.28),
                 position=(0,4.6,12),color=color.rgb32(65,40,18))
        # Kapı tahtaları
        for dy in [-0.8,0,0.8]:
            self._mk(model='cube',scale=(2.6,0.55,0.18),
                     position=(0,2.25+dy,12.01),
                     color=color.rgb32(90,55,25))
        # Tabela
        sign=self._mk(model='cube',scale=(2.2,0.5,0.12),
                      position=(0,3.7,12.05),
                      color=color.rgb32(140,90,35))
        Text('KALEYE GİT ',parent=sign,scale=6,
             origin=(0,0),color=color.rgb32(255,220,80),z=-6)

    def update(self):
        self.elapsed += time.dt
        for flame, inner, light, base_y, phase in self.torch_flames:
            pulse = math.sin(self.elapsed * 9.0 + phase)
            flame.y = base_y + pulse * 0.045
            inner.y = flame.y + 0.06
            flame.scale = (0.24 + 0.025 * pulse, 0.34 + 0.06 * pulse, 0.24 + 0.025 * pulse)
            flame.rotation_z = math.sin(self.elapsed * 6.0 + phase) * 9.0
            inner.scale_y = 0.12 + 0.025 * math.sin(self.elapsed * 12.0 + phase)
            light['intensity'] = 1.35 + 0.60 * (0.5 + 0.5 * pulse)
        # Altyazı açıkken dünya güncellenmeye devam eder; yalnız etkileşimler kilitlenir.
        if not g.dlg_open and not g.has_sword:
            if distance_xz(player.position,self.sword.position)<2.2:
                hud.set_hint('[ E ]  Kılıcı Al')
                if held_keys['e']:
                    g.has_sword=True; self.sword.visible=False
                    give_sword()
                    hud.refresh(); hud.set_hint('')
                    dlg.show("Eski ama keskin bir kılıç!\n\nSOL TIK ile savurabilirsin.\nHer vuruş düşmandan 5 HP götürür.\n\nŞimdi kaleye ilerle.",'Kılıç Bulundu')
                return
            else: hud.set_hint('')
        # Kapı
        d=distance_xz(player.position,Vec3(0,2,12))
        if not g.dlg_open and d<3:
            if g.has_sword:
                hud.set_hint('[ E ]  Kaleye Gir')
                if held_keys['e']: hud.set_hint(''); load_chapter(2)
            else: hud.set_hint('Önce kılıcı bul!')
        elif g.has_sword: hud.set_hint('')

    def clear(self):
        for e in self.ents:
            try: destroy(e)
            except: pass
        scene.fog_density=0

# ─────────────────────────────────────────────────────────────
#  BÖLÜM 2 – ŞİFRELİ ŞATO / KEŞİF BULMACASI
# ─────────────────────────────────────────────────────────────
class Chapter2(Entity):
    """Ormandan sonraki şato. Dört odadaki çekmeceler dört haneli kodu verir."""
    PASSWORD = '4719'

    def __init__(self):
        super().__init__()
        self.ents = []
        self.drawers = []
        self.torch_flames = []
        self.found_digits = {}
        self.code_mode = False
        self.typed_code = ''
        self.door_opening = False
        self.elapsed = 0.0
        self.atmosphere_fog_color = color.rgb32(15, 15, 28)
        self.atmosphere_fog_density = 0.018
        self._build()
        self._update_note()
        hud.code_note.enabled = True
        hud.code_note_bg.enabled = True
        invoke(lambda: dlg.show(
            "Ormandaki kapının ardında eski bir şato buldun.\n\n"
            "Boss'a giden ana kapı dört haneli bir şifreyle kilitli.\n"
            "Dört odadaki çekmeceleri E ile aç ve içlerindeki sayı kâğıtlarını bul.\n"
            "Bulduğun sayılar ekranın sağ üst köşesinde sırayla not edilecek.\n\n"
            "Dört sayıyı da bulunca ana kapının yanındaki panele git.",
            'Bölüm 2 – Şifreli Şato'), delay=0.7)

    def _mk(self, *a, **kw):
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e = Entity(*a, **kw)
        self.ents.append(e)
        try:
            sc=e.scale
            dims=(abs(float(sc.x)),abs(float(sc.y)),abs(float(sc.z)))
            e._kk_lod_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 3.6 and max(dims) > 0.015)
            e._kk_occlusion_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 12.0 and max(dims) > 0.015)
        except Exception:
            e._kk_lod_candidate=False
            e._kk_occlusion_candidate=False
        return e

    def _wall(self, pos, scale):
        return self._mk(model='cube', position=pos, scale=scale,
                        color=color.rgb32(58, 49, 53), collider='box')

    def _build_room_furniture(self, room_name, x, z, digit, slot):
        # Oda zemini ve halı; odalar açık kapı boşluklarıyla merkez koridora bağlanır.
        carpet = self._mk(model='cube', position=(x, 0.015, z),
                          scale=(7.2, 0.04, 6.0),
                          color=color.rgb32(62, 25, 36), collider=None)
        # Yatak: ahşap iskelet, koyu yatak ve yastık.
        self._mk(model='cube', position=(x + 1.35, 0.38, z + 1.45),
                 scale=(2.0, 0.42, 3.0), color=color.rgb32(76, 43, 29), collider='box')
        self._mk(model='cube', position=(x + 1.35, 0.64, z + 1.45),
                 scale=(1.82, 0.16, 2.78), color=color.rgb32(71, 42, 66), collider=None)
        self._mk(model='cube', position=(x + 1.35, 0.76, z + 2.42),
                 scale=(1.35, 0.14, 0.53), color=color.rgb32(210, 197, 180), collider=None)
        # Dolap, kapak çizgileri ve kulplar.
        self._mk(model='cube', position=(x - 2.0, 1.22, z + 1.7),
                 scale=(1.3, 2.45, 0.85), color=color.rgb32(70, 43, 27), collider='box')
        for dx in (-0.32, 0.32):
            self._mk(model='cube', position=(x - 2.0 + dx, 1.24, z + 1.235),
                     scale=(0.035, 2.15, 0.045), color=color.rgb32(135, 93, 45), collider=None)
            self._mk(model='sphere', position=(x - 2.0 + dx, 1.22, z + 1.16),
                     scale=0.09, color=color.rgb32(205, 165, 75), unlit=True, shader=None, collider=None)
        # Komodin ve şamdan.
        self._mk(model='cube', position=(x + 2.45, 0.48, z - 0.35),
                 scale=(0.9, 0.9, 0.8), color=color.rgb32(83, 51, 31), collider='box')
        self._mk(model=cyl(), position=(x + 2.45, 1.12, z - 0.35),
                 scale=(0.08, 0.45, 0.08), color=color.rgb32(170, 130, 65), collider=None)
        flame = self._mk(model='sphere', position=(x + 2.45, 1.42, z - 0.35),
                         scale=(0.105, 0.26, 0.105), color=color.rgb32(255, 118, 26),
                         unlit=True, shader=None, collider=None)
        inner_flame = self._mk(model='sphere', position=(x + 2.45, 1.45, z - 0.35),
                                scale=(0.055, 0.16, 0.055), color=color.rgb32(255, 220, 120),
                                unlit=True, shader=None, collider=None)
        mark_glow(flame)
        warm_source = register_warm_light_at(self, (x + 2.45, 1.5, z - 0.35),
                                             tint=(1.0, 0.45, 0.12), radius=5.5, intensity=0.85)
        self.torch_flames.append((flame, inner_flame, warm_source, 1.42, random.uniform(0, math.tau)))
        # Çekmeceli şifonyer. Ön yüz ayrı Entity olduğundan açıldığında dışarı kayar.
        cabinet_x = x - 0.25
        cabinet_z = z - 1.75
        self._mk(model='cube', position=(cabinet_x, 0.58, cabinet_z),
                 scale=(1.65, 1.12, 0.88), color=color.rgb32(88, 52, 30), collider='box')
        # Her odada tek, gerçekten etkileşilebilir bir çekmece bulunur.
        # Böylece görünen hiçbir çekmece sahte kalmaz: E ile açılınca kâğıt çıkar.
        drawer_y = 0.58
        front = self._mk(model='cube',
                         position=(cabinet_x, drawer_y, cabinet_z - 0.48),
                         scale=(1.42, 0.62, 0.10),
                         color=color.rgb32(120, 76, 42), collider=None)
        handle = self._mk(model='sphere',
                          position=(cabinet_x, drawer_y, cabinet_z - 0.55),
                          scale=(0.10, 0.07, 0.06),
                          color=color.rgb32(220, 181, 91), unlit=True, shader=None, collider=None)
        self.drawers.append({
            'room': room_name, 'slot': slot, 'digit': str(digit),
            'position': Vec3(cabinet_x, 0.5, cabinet_z - 1.05),
            'front': front, 'handle': handle, 'opened': False,
            'paper': None, 'note': None
        })
        # Odanın adı, duvarda küçük bir levha olarak bulunur.
        plaque = self._mk(model='cube', position=(x, 2.35, z - 2.78),
                          scale=(2.35, 0.42, 0.08), color=color.rgb32(100, 70, 42), collider=None)
        Text(room_name.upper(), parent=plaque, scale=4.5, origin=(0, 0),
             color=color.rgb32(245, 205, 125), z=-2)

    def _build(self):
        self._mk(model='sphere', scale=500, color=color.rgb32(3, 4, 13),
                 double_sided=True, unlit=True, shader=None, collider=None)
        scene.fog_color = self.atmosphere_fog_color
        scene.fog_density = self.atmosphere_fog_density
        add_night_sky(self, include_moon=False, star_count=35)
        # Zemin
        for gx in range(-8, 9):
            for gz in range(-7, 8):
                tile = self._mk(model='cube', position=(gx * 2, -0.12, gz * 2),
                                scale=(2.02, 0.24, 2.02),
                                color=color.rgb32(58 + random.randint(0, 8),
                                                  51 + random.randint(0, 7),
                                                  55 + random.randint(0, 8)),
                                collider='box')
                register_reflection_surface(tile)
        # Dış duvarlar. Kuzeyde ortada ana kapı boşluğu, güneyde şatoya giriş boşluğu.
        self._wall((-15.4, 3.5, 0), (0.8, 7, 29))
        self._wall((15.4, 3.5, 0), (0.8, 7, 29))
        self._wall((-8.25, 3.5, 14.4), (13.5, 7, 0.8))
        self._wall((8.25, 3.5, 14.4), (13.5, 7, 0.8))
        self._wall((-8.25, 3.5, -14.4), (13.5, 7, 0.8))
        self._wall((8.25, 3.5, -14.4), (13.5, 7, 0.8))
        # Dört gerçek oda: bölme duvarları birbirine bağlanır; her oda için
        # yaklaşık 2-3 birimlik geçiş boşluğu bırakılır. Merkezde küçük bir kesişim vardır.
        for x, width in [(-12.0, 6.0), (-4.0, 6.0), (4.0, 6.0), (12.0, 6.0)]:
            self._wall((x, 3.25, 0), (width, 6.5, 0.45))
        for z, length in [(-11.5, 5.0), (-3.5, 5.0), (3.5, 5.0), (11.5, 5.0)]:
            self._wall((0, 3.25, z), (0.45, 6.5, length))

        # Kapalı taş tavan. Kirişler tavanı daha inandırıcı gösterir ve dışarı düşmeyi önler.
        self._wall((0, 7.15, 0), (31.5, 0.75, 29.7))
        for bx in (-10.0, -5.0, 0.0, 5.0, 10.0):
            self._wall((bx, 6.68, 0), (0.28, 0.22, 28.5))
        for bz in (-9.0, -3.0, 3.0, 9.0):
            self._wall((0, 6.68, bz), (30.5, 0.22, 0.28))
        # Ana kapı, tuş paneli ve çerçeve.
        for sx in (-1.65,  1.65):
            self._mk(model='cube', position=(sx, 2.35, 14.0), scale=(0.32, 4.7, 0.6),
                     color=color.rgb32(65, 37, 23), collider='box')
        self._mk(model='cube', position=(0, 4.75, 14.0), scale=(3.55, 0.32, 0.6),
                 color=color.rgb32(65, 37, 23), collider='box')
        self.exit_door = self._mk(model='cube', position=(0, 2.25, 14.15),
                                  scale=(2.75, 4.3, 0.28), color=color.rgb32(83, 43, 24),
                                  collider='box')
        for i in range(5):
            self._mk(model='cube', position=(-1.0 + i * 0.5, 2.2, 13.98),
                     scale=(0.045, 3.85, 0.06), color=color.rgb32(110, 64, 35), collider=None)
        self._mk(model='cube', position=(0, 3.4, 13.82), scale=(0.42, 0.42, 0.1),
                 color=color.rgb32(205, 161, 63), collider=None)
        # Animasyonlu duvar meşaleleri: iki kat alev, emissive parıltı ve dinamik ışık.
        for tx, tz in [(-13, 0), (13, 0), (-4, 11), (4, 11), (-4, -11), (4, -11)]:
            self._mk(model='cylinder', position=(tx, 2.1, tz), scale=(0.10, 1.4, 0.10),
                     color=color.rgb32(70, 45, 27), collider=None)
            fire = self._mk(model='sphere', position=(tx, 2.9, tz), scale=(0.17, 0.38, 0.17),
                            color=color.rgb32(255, 112, 24), unlit=True, shader=None, collider=None)
            inner = self._mk(model='sphere', position=(tx, 2.92, tz), scale=(0.085, 0.24, 0.085),
                             color=color.rgb32(255, 221, 120), unlit=True, shader=None, collider=None)
            mark_glow(fire)
            light_src = register_warm_light_at(self, (tx, 2.9, tz), tint=(1.0, 0.4, 0.12), radius=8.0, intensity=1.25)
            self.torch_flames.append((fire, inner, light_src, 2.9, random.uniform(0, math.tau)))
        self._build_room_furniture('KUZEYBATI ODASI', -8.2, 7.5, 4, 1)
        self._build_room_furniture('KUZEYDOĞU ODASI', 8.2, 7.5, 7, 2)
        self._build_room_furniture('GÜNEYBATI ODASI', -8.2, -7.5, 1, 3)
        self._build_room_furniture('GÜNEYDOĞU ODASI', 8.2, -7.5, 9, 4)
        # Kapı yanındaki küçük kod paneli.
        panel = self._mk(model='cube', position=(2.15, 1.65, 13.65), scale=(0.28, 0.72, 0.15),
                         color=color.rgb32(36, 43, 51), collider=None)
        for i in range(4):
            self._mk(model='cube', position=(2.15, 1.86 - i * 0.14, 13.54),
                     scale=(0.12, 0.07, 0.035), color=color.rgb32(65, 195, 170),
                     unlit=True, shader=None, collider=None)
        register_warm_light_at(self, (2.15, 1.7, 13.4), tint=(0.10, 0.75, 0.55), radius=4.0, intensity=0.6)
        # Ekran üstü şifre paneli. Dünya culling listesine eklenmez.
        self.code_overlay = Entity(parent=camera.ui, enabled=False, z=-0.7)
        Entity(parent=self.code_overlay, model='quad', scale=(0.80, 0.56),
               color=color.rgba32(5, 9, 17, 245), z=0)
        Entity(parent=self.code_overlay, model='quad', scale=(0.77, 0.52),
               color=color.rgba32(20, 30, 45, 250), z=-0.01)
        Entity(parent=self.code_overlay, model='quad', scale=(0.74, 0.012), y=0.20,
               color=color.rgb32(70, 210, 185), z=-0.02)
        self.code_title = Text('ŞATO ANA KAPISI', parent=self.code_overlay, y=0.14,
                               origin=(0, 0), scale=1.4,
                               color=color.rgb32(120, 245, 210), z=-0.03)
        self.code_value_text = Text('____', parent=self.code_overlay, y=0.015,
                                    origin=(0, 0), scale=2.2,
                                    color=color.rgb32(255, 220, 145), z=-0.03)
        self.code_help = Text('Rakamları yaz  |  ENTER: Onayla  |  BACKSPACE: Sil  |  ESC: Kapat',
                              parent=self.code_overlay, y=-0.15, origin=(0, 0), scale=0.72,
                              color=color.rgb32(190, 205, 220), z=-0.03)

    def _update_note(self):
        lines = ['ŞİFRE KÂĞITLARI']
        for slot in range(1, 5):
            digit = self.found_digits.get(slot, '_')
            lines.append(f'{slot}. oda: {digit}')
        hud.code_note.text = '\n'.join(lines)

    def _refresh_code(self):
        self.code_value_text.text = (self.typed_code + '_' * (4 - len(self.typed_code)))[:4]

    def _nearest_drawer(self):
        nearest = None
        best = 999.0
        for drawer in self.drawers:
            if drawer['opened']:
                continue
            d = distance_xz(player.position, drawer['position'])
            if d < best:
                best = d
                nearest = drawer
        return nearest, best

    def interact(self):
        if self.code_mode or self.door_opening or g.dlg_open:
            return
        drawer, dist = self._nearest_drawer()
        if drawer and dist < 2.5:
            self._open_drawer(drawer)
            return
        if distance_xz(player.position, Vec3(0, 2, 13.6)) < 3.4:
            if len(self.found_digits) < 4:
                hud.set_hint(f'Şifre eksik: {4 - len(self.found_digits)} çekmece daha bulmalısın.')
                hud.announce('Dört odadaki şifre kâğıtlarını bulmalısın.', 2.5)
                return
            self.code_mode = True
            self.typed_code = ''
            self._refresh_code()
            self.code_overlay.enabled = True
            player.speed = 0
            player.jump_height = 0
            mouse.locked = False
            mouse.visible = True
            hud.set_hint('')
            return
        if drawer and dist < 4.5:
            hud.set_hint(f'[ E ] {drawer["room"]}: Çekmeceyi aç')
        else:
            hud.set_hint('[ E ] Yakındaki çekmeceyi veya ana kapıyı kullan')

    def _open_drawer(self, drawer):
        drawer['opened'] = True
        front = drawer['front']
        handle = drawer['handle']
        front.animate_position(front.position + Vec3(0, 0, -0.34), duration=0.22)
        handle.animate_position(handle.position + Vec3(0, 0, -0.34), duration=0.22)
        digit = drawer['digit']
        paper = self._mk(model='cube', position=(drawer['position'].x, 0.86, drawer['position'].z + 0.08),
                         scale=(0.43, 0.025, 0.48), color=color.rgb32(235, 224, 190), collider=None)
        drawer['paper'] = paper
        # Bulunan şifre kâğıdı ve rakamı hiçbir zaman LOD tarafından atılmasın.
        paper._kk_lod_candidate = False
        paper._kk_occlusion_candidate = True
        drawer['note'] = Text(f'{drawer["slot"]}: {digit}', parent=paper, scale=4.8,
                              origin=(0, 0), color=color.rgb32(37, 28, 23), z=-2)
        self.found_digits[drawer['slot']] = digit
        self._update_note()
        hud.refresh()
        hud.announce(f'{drawer["room"].title()}: kâğıtta {digit} sayısı var.', 2.8)
        hud.set_hint(f'Şifre kâğıdı bulundu: {drawer["slot"]}. sıradaki rakam {digit}')
        if len(self.found_digits) == 4:
            hud.announce('Dört kâğıt da bulundu! Kapının yanındaki panele git.', 4.0)

    def handle_input(self, key):
        if not self.code_mode:
            return False
        if key in ('escape',):
            self.code_mode = False
            self.code_overlay.enabled = False
            player.speed = 6
            player.jump_height = 2.4
            mouse.locked = True
            mouse.visible = False
            return True
        if key in ('backspace', 'delete'):
            self.typed_code = self.typed_code[:-1]
            self._refresh_code()
            return True
        if key in ('enter', 'numpad enter'):
            if self.typed_code == self.PASSWORD and len(self.found_digits) == 4:
                self.code_mode = False
                self.door_opening = True
                self.code_title.text = 'ŞİFRE DOĞRU'
                self.code_value_text.text = 'KAPI AÇILIYOR'
                self.code_help.text = 'Karanlık Şövalye seni bekliyor...'
                self.exit_door.collider = None
                self.exit_door.animate_rotation_y(95, duration=0.65)
                invoke(self._leave_to_boss, delay=1.25)
            else:
                self.code_title.text = 'ŞİFRE YANLIŞ'
                self.typed_code = ''
                self._refresh_code()
                invoke(lambda: setattr(self.code_title, 'text', 'ŞATO ANA KAPISI'), delay=1.2)
            return True
        if key in tuple(str(i) for i in range(10)) and len(self.typed_code) < 4:
            self.typed_code += key
            self._refresh_code()
            return True
        return True

    def _leave_to_boss(self):
        self.code_overlay.enabled = False
        player.speed = 6
        player.jump_height = 2.4
        mouse.locked = True
        mouse.visible = False
        hud.code_note.enabled = False
        hud.code_note_bg.enabled = False
        load_chapter(3)

    def update(self):
        self.elapsed += time.dt
        for flame, inner, light_source, base_y, phase in self.torch_flames:
            pulse = math.sin(self.elapsed * 11.0 + phase)
            flame.y = base_y + 0.035 * pulse
            flame.scale = (0.17 + 0.018 * pulse, 0.36 + 0.045 * pulse, 0.17 + 0.018 * pulse)
            flame.rotation_z = math.sin(self.elapsed * 7.5 + phase) * 8.0
            inner.y = flame.y + 0.025
            inner.scale_y = 0.22 + 0.025 * math.sin(self.elapsed * 13.0 + phase + 0.7)
            try:
                light_source['node'].position = flame.position
                light_source['intensity'] = 0.95 + 0.36 * (0.5 + 0.5 * pulse)
            except Exception:
                pass
        if self.code_mode or self.door_opening:
            hud.set_hint('')
            return
        if g.dlg_open:
            return
        drawer, dist = self._nearest_drawer()
        door_dist = distance_xz(player.position, Vec3(0, 2, 13.6))
        if drawer and dist < 2.5:
            hud.set_hint(f'[ E ] {drawer["room"].title()}: Çekmeceyi aç')
        elif door_dist < 3.4:
            if len(self.found_digits) < 4:
                hud.set_hint(f'Ana kapı kilitli. {4-len(self.found_digits)} şifre kâğıdı eksik.')
            else:
                hud.set_hint('[ E ] Şifre panelini kullan')
        else:
            hud.set_hint('')

    def clear(self):
        hud.code_note.enabled = False
        hud.code_note_bg.enabled = False
        if getattr(self, 'code_overlay', None):
            destroy(self.code_overlay)
        for e in self.ents:
            try:
                destroy(e)
            except Exception:
                pass
        scene.fog_density = 0


# ─────────────────────────────────────────────────────────────
#  BÖLÜM 3 – KALE BOSS ARENASI
# ─────────────────────────────────────────────────────────────
class Chapter3(Entity):
    def __init__(self):
        super().__init__()
        self.ents=[]; self.boss_alive=True
        self.torch_flames=[]
        self.elapsed=0.0
        self.atmosphere_fog_color = color.rgb32(10, 12, 27)
        self.atmosphere_fog_density = 0.030
        self.boss_max=100; self.boss_hp_val=100
        self.state='idle'; self.timer=0.0; self.cool=0.0; self.enraged=False
        self._build()
        invoke(lambda: dlg.show(
            "Şifresi çözülmüş ana kapıdan geçip savaş salonuna ulaştın.\n\n"
            "Anahtarı bul, Karanlık Şövalye'yi yen,\n"
            "çıkış kapısını aç ve lav parkuruna ilerle!\n\n"
            "Şövalye seni görünce üstüne gelir.\n"
            "Yakınındayken SOL TIK ile kılıcını salla.\n"
            "Darbesini hazırlarken geri çekilirsen kaçarsın!",
            'Bölüm 3 – Karanlık Kale'), delay=0.6)

    def _mk(self,*a,**kw):
        # Dünya modellerinde gölge alabilen ışıklandırma shader'ı.
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e=Entity(*a,**kw); self.ents.append(e)
        try:
            sc=e.scale
            dims=(abs(float(sc.x)),abs(float(sc.y)),abs(float(sc.z)))
            # Uzakta pahalı olmayan, küçük dekoratif ayrıntılar LOD ile render dışına alınabilir.
            # Collider taşıyan duvar/zemin/mobilya ve çok büyük gök kubbeleri korunur.
            e._kk_lod_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 3.6 and max(dims) > 0.015)
            e._kk_occlusion_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 12.0 and max(dims) > 0.015)
        except Exception:
            e._kk_lod_candidate=False
            e._kk_occlusion_candidate=False
        return e

    def _make_torch(self, x, z, base_y=0.0, pole_h=1.15):
        self._mk(model=cyl(), scale=(0.09, pole_h, 0.09),
                 position=(x, base_y + pole_h / 2, z),
                 color=color.rgb32(60, 36, 18))
        self._mk(model='cube', scale=(0.24, 0.06, 0.08),
                 position=(x, base_y + pole_h - 0.05, z),
                 color=color.rgb32(75, 42, 19))
        flame_y = base_y + pole_h + 0.08
        flame = self._mk(model='sphere', scale=0.28,
                         position=(x, flame_y, z),
                         color=color.rgb32(255, 105, 20), unlit=True, shader=None)
        inner = self._mk(model='sphere', scale=0.14,
                         position=(x, flame_y + 0.06, z),
                         color=color.rgb32(255, 198, 80), unlit=True, shader=None)
        mark_glow(flame)
        light = register_warm_light(flame, tint=(1.0, 0.34, 0.06), radius=12.5, intensity=1.9)
        add_warm_ground_glow(self, x, z, floor_y=base_y + 0.012,
                             radius=2.1 if base_y < 1.0 else 0.95, tint=(255, 112, 28))
        self.torch_flames.append((flame, inner, light, flame_y, random.uniform(0, math.tau)))

    def _build(self):
        # Açık kale avlusu: ay ve yıldızlar görülebilsin.
        self._mk(model='sphere',scale=500,
                 color=color.rgb32(3, 4, 14),double_sided=True, unlit=True, shader=None)
        add_night_sky(self, include_moon=True, star_count=90)
        self.atmosphere_fog_color = color.rgb32(10, 12, 27)
        self.atmosphere_fog_density = 0.030
        scene.fog_color=self.atmosphere_fog_color
        scene.fog_density=self.atmosphere_fog_density
        add_low_mist(self, [(-8, -6, 4.5), (7, -5, 4.0), (-5, 7, 3.5),
                            (5, 8, 4.0)], (34, 40, 72))

        # Zemin – taş mozaik
        for gx in range(-6,7):
            for gz in range(-6,7):
                dark = random.randint(0,1)
                c_ = color.rgb32(48+dark*8,45+dark*8,55+dark*8)
                floor_tile = self._mk(model='cube', collider='box',
                         position=(gx*2,-0.1,gz*2),
                         scale=(2.02,0.2,2.02), color=c_, shader=None if reflection_enabled else torch_lighting_shader)
                register_reflection_surface(floor_tile)

        # Kale duvarları
        for pos,scl in [
            ((0,3.5,-12),(24,7,1.4)),
            ((0,3.5, 12),(24,7,1.4)),
            ((-12,3.5,0),(1.4,7,24)),
            (( 12,3.5,0),(1.4,7,24)),
        ]:
            self._mk(model='cube',position=pos,scale=scl,
                     color=color.rgb32(52,48,58),collider='box')

        # Üstü kapalı savaş salonu: çatı ve kirişler dış ortamdan kopuk, ağır taş kale hissi verir.
        self._mk(model='cube', position=(0, 7.28, 0), scale=(24.4, 0.72, 24.4),
                 color=color.rgb32(38, 35, 44), collider='box')
        for bx in (-9, -4.5, 0, 4.5, 9):
            self._mk(model='cube', position=(bx, 6.78, 0), scale=(0.34, 0.22, 23.5),
                     color=color.rgb32(30, 28, 36), collider=None)
        for bz in (-9, -4.5, 0, 4.5, 9):
            self._mk(model='cube', position=(0, 6.78, bz), scale=(23.5, 0.22, 0.34),
                     color=color.rgb32(30, 28, 36), collider=None)

        # Mazgallar
        for i in range(-5,6,2):
            for z in [-11.3,11.3]:
                self._mk(model='cube',position=(i*2,7.0,z),
                         scale=(1.2,1.4,1.5),
                         color=color.rgb32(42,38,48))

        # Sütunlar
        for px,pz in [(-8,8),(-8,-8),(8,8),(8,-8),(-8,0),(8,0),(0,8),(0,-8)]:
            self._mk(model=cyl(),scale=(0.55,5.5,0.55),
                     position=(px,2.75,pz),
                     color=color.rgb32(62,58,70))
            self._mk(model='cube',scale=(0.85,0.28,0.85),
                     position=(px,5.64,pz),
                     color=color.rgb32(72,68,80))

        # Zemin meşaleleri: tabanları zemine oturur; boss arenasının ortasında havada durmazlar.
        for tx,tz in [(-9,-6),(9,-6),(-9,6),(9,6)]:
            self._make_torch(tx, tz, base_y=0.0, pole_h=1.2)

        # Üst meşaleler: gerçekten sütun başlıklarının üzerine oturtulmuştur.
        # Sütun üst yüzeyi yaklaşık Y=5.78 olduğundan direk tam oradan başlar.
        for tx,tz in [(-8,-8),(8,-8),(-8,8),(8,8)]:
            self._make_torch(tx, tz, base_y=5.55, pole_h=0.55)

        # === ANAHTAR ===
        self.key = self._mk(
            model='cube',scale=(0.12,0.48,0.07),
            position=(-6,0.24,6),collider='box',
            color=color.rgb32(255,200,0))
        self._mk(model='cube',scale=(0.2,0.08,0.07),
                 position=(-6,0.41,6),
                 color=color.rgb32(255,200,0))
        glow=self._mk(model='sphere',scale=0.55,
                      position=(-6,0.46,6),
                      color=color.rgba32(255,210,0,50))
        mark_glow(glow)
        register_warm_light(glow, tint=(1.0, 0.72, 0.12), radius=8.0, intensity=1.65)
        add_warm_ground_glow(self, -6, 6, floor_y=0.018, radius=1.45, tint=(255, 186, 35))

        # === BOSS ===
        self.boss = self._mk(position=(0,0,8))
        def bm(**kw):
            kw.setdefault('shader', torch_lighting_shader)
            return Entity(parent=self.boss, **kw)
        self.parts = []
        def body_part(**kw):
            e = bm(**kw); self.parts.append(e); return e
        self.boss_body = body_part(model='cube', scale=(1.5,2.6,1.5), y=1.3,
                                   collider='box')
        body_part(model='sphere', scale=1.15, y=3.0)
        for sx in (-0.9, 0.9):
            body_part(model='sphere', scale=0.7, x=sx, y=2.4)
        for ex in (-0.25, 0.25):
            bm(model='sphere', scale=0.2, position=(ex,3.04,0.55),
               color=color.rgb32(255,40,0))
        # Sol kol
        self.arm_l = bm(position=(-1.0,2.4,0))
        Entity(parent=self.arm_l, model='cube', scale=(0.5,1.7,0.5), y=-0.85)
        self.parts.append(self.arm_l.children[0])
        # Sağ kol + gürz
        self.arm_r = bm(position=(1.0,2.4,0))
        arm_mesh = Entity(parent=self.arm_r, model='cube', scale=(0.5,1.7,0.5), y=-0.85)
        self.parts.append(arm_mesh)
        Entity(parent=self.arm_r, model='cube', scale=(0.9,0.9,0.9),
               y=-1.9, z=0.25, color=color.rgb32(55,55,62))
        self._tint(self._cur_color())

        # === KUZEY KAPISI ===
        for sx in [-1.4,1.4]:
            self._mk(model='cube',scale=(0.28,5.2,0.28),
                     position=(sx,2.6,-11.4),
                     color=color.rgb32(60,40,18))
        self._mk(model='cube',scale=(3.1,0.3,0.28),
                 position=(0,5.3,-11.4),
                 color=color.rgb32(60,40,18))
        self.exit_door=self._mk(
            model='cube',scale=(2.6,4.9,0.22),
            position=(0,2.5,-11.4),collider='box',
            color=color.rgb32(55,38,18))
        Text('SON BÖLÜM ',parent=self.exit_door,
             scale=5.5,origin=(0,0),
             color=color.rgb32(255,175,40),z=-6)
        register_warm_light_at(self, (0, 3.0, -11), tint=(1.0, 0.55, 0.12), radius=10.0, intensity=1.5)
        add_warm_ground_glow(self, 0, -10.7, floor_y=0.02, radius=1.8, tint=(255, 150, 32))

    # ── Boss renkleri ────────────────────────────────────────
    def _cur_color(self):
        if self.state == 'windup':
            return color.rgb32(255,110,20)
        return color.rgb32(175,22,12) if self.enraged else color.rgb32(110,15,15)

    def _tint(self, c):
        for p in self.parts:
            p.color = c

    # ── Oyuncunun saldırısı (sol tık) ────────────────────────
    def on_player_attack(self):
        if not self.boss_alive: return
        if distance_xz(player.position, self.boss.position) > 3.6: return
        self.damage_boss(5)

    def damage_boss(self, amount):
        self.boss_hp_val = max(0, self.boss_hp_val - amount)
        if self.state == 'idle':
            self.state = 'chase'
        hud.set_boss(self.boss_hp_val / self.boss_max)
        hud.popup(f'-{amount}')
        # Beyaz yanıp sönme + hafif geri itme
        self._tint(color.rgb32(255,235,235))
        invoke(self._restore_color, delay=0.1)
        away = Vec3(self.boss.x - player.x, 0, self.boss.z - player.z)
        if away.length() > 0.01:
            away = away.normalized()
            self.boss.x = max(-10, min(10, self.boss.x + away.x * 0.45))
            self.boss.z = max(-10, min(10, self.boss.z + away.z * 0.45))
        if not self.enraged and self.boss_hp_val <= self.boss_max * 0.5:
            self.enraged = True
            hud.announce('Şövalye öfkelendi!')
        if self.boss_hp_val <= 0:
            self._boss_die()

    def _restore_color(self):
        if self.boss_alive:
            self._tint(self._cur_color())

    def _boss_die(self):
        self.boss_alive = False
        g.enemy_dead = True
        self.boss_body.collider = None
        hud.set_boss(None)
        hud.set_hint('')
        self.boss.animate_scale(Vec3(0.01,0.01,0.01), duration=0.7)
        destroy(self.boss, delay=0.8)
        invoke(lambda: dlg.show(
            "Karanlık Şövalye yere yıkıldı!\nKale titredi.\n\nŞimdi çıkış kapısını aç!",
            'Boss Yenildi'), delay=0.9)

    # ── Boss yapay zekâsı ────────────────────────────────────
    def _boss_ai(self, dt):
        b = self.boss
        to = Vec3(player.x - b.x, 0, player.z - b.z)
        dist = to.length()
        if dist > 0.01:
            b.rotation_y = math.degrees(math.atan2(to.x, to.z))
        self.cool = max(0.0, self.cool - dt)
        speed = 4.6 if self.enraged else 3.2
        dmg   = 12  if self.enraged else 8
        wind  = 0.45 if self.enraged else 0.65

        if self.state == 'idle':
            if dist < 11:
                self.state = 'chase'
                hud.set_boss(self.boss_hp_val / self.boss_max)
                hud.announce('Karanlık Şövalye seni fark etti!')
        elif self.state == 'chase':
            if dist > 2.6:
                d = to.normalized()
                b.x = max(-10, min(10, b.x + d.x * speed * dt))
                b.z = max(-10, min(10, b.z + d.z * speed * dt))
            elif self.cool <= 0:
                self.state = 'windup'
                self.timer = wind
                self.arm_r.animate('rotation_x', 160, duration=wind)
                self._tint(self._cur_color())
        elif self.state == 'windup':
            self.timer -= dt
            if self.timer <= 0:
                self.arm_r.animate('rotation_x', 0, duration=0.15)
                if dist < 3.3:
                    hurt_player(dmg)
                self.state = 'recover'
                self.timer = 0.5
                self._tint(self._cur_color())
        elif self.state == 'recover':
            self.timer -= dt
            if self.timer <= 0:
                self.state = 'chase'
                self.cool = 0.7 if self.enraged else 1.0

    def update(self):
        self.elapsed += time.dt
        for flame, inner, light, base_y, phase in self.torch_flames:
            pulse = math.sin(self.elapsed * 8.5 + phase)
            flame.y = base_y + pulse * 0.045
            inner.y = flame.y + 0.06
            flame.scale = (0.24 + 0.025 * pulse, 0.34 + 0.06 * pulse, 0.24 + 0.025 * pulse)
            flame.rotation_z = math.sin(self.elapsed * 6.0 + phase) * 9.0
            inner.scale_y = 0.12 + 0.025 * math.sin(self.elapsed * 12.0 + phase)
            light['intensity'] = 1.45 + 0.65 * (0.5 + 0.5 * pulse)
        # Boss AI devam ederken altyazı yalnız oyuncu etkileşimlerini engeller.
        dt = time.dt

        if self.boss_alive:
            self._boss_ai(dt)
            if g.dlg_open: return

        hint = ''
        # Anahtar
        if not g.has_key:
            if distance_xz(player.position, self.key.position) < 2.2:
                hint = '[ E ]  Anahtarı Al'
                if held_keys['e']:
                    g.has_key = True; self.key.visible = False
                    hud.refresh(); hud.set_hint('')
                    dlg.show("Altın anahtar!\nKaranlık Şövalye'yi yenip\nçıkış kapısını açabilirsin.", 'Anahtar Bulundu')
                    return
        # Boss yakınında mı?
        if self.boss_alive and distance_xz(player.position, self.boss.position) < 3.6:
            hint = '[ SOL TIK ]  Saldır!'
        # Kapı
        if distance_xz(player.position, Vec3(0,2.5,-11.4)) < 3:
            if g.has_key and not self.boss_alive:
                hint = '[ E ]  Kapıyı Aç'
                if held_keys['e']: hud.set_hint(''); load_chapter(4); return
            elif self.boss_alive: hint = 'Boss hâlâ yaşıyor!'
            elif not g.has_key:   hint = 'Anahtara ihtiyacın var!'
        hud.set_hint(hint)

    def clear(self):
        hud.set_boss(None)
        for e in self.ents:
            try: destroy(e)
            except: pass
        scene.fog_density=0

# ─────────────────────────────────────────────────────────────
#  BÖLÜM 4 – EJDERHA YUVASI / LAV PARKURU
# ─────────────────────────────────────────────────────────────
class Chapter4(Entity):
    def __init__(self):
        super().__init__()
        self.ents=[]
        self.atmosphere_fog_color = color.rgb32(48, 11, 3)
        self.atmosphere_fog_density = 0.034
        self._build()
        invoke(lambda: dlg.show(
            "Ejderhanın yuvasındasın!\n\n"
            "Lava dokunmadan platformları geç\n"
            "ve Prenses'i kafesten kurtar.",
            'Bölüm 4 – Ejderha Yuvası'), delay=0.6)

    def _mk(self,*a,**kw):
        # Dünya modellerinde gölge alabilen ışıklandırma shader'ı.
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e=Entity(*a,**kw); self.ents.append(e)
        try:
            sc=e.scale
            dims=(abs(float(sc.x)),abs(float(sc.y)),abs(float(sc.z)))
            # Uzakta pahalı olmayan, küçük dekoratif ayrıntılar LOD ile render dışına alınabilir.
            # Collider taşıyan duvar/zemin/mobilya ve çok büyük gök kubbeleri korunur.
            e._kk_lod_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 3.6 and max(dims) > 0.015)
            e._kk_occlusion_candidate=(kw.get('collider') is None and bool(kw.get('visible', True)) and max(dims) <= 12.0 and max(dims) > 0.015)
        except Exception:
            e._kk_lod_candidate=False
            e._kk_occlusion_candidate=False
        return e

    def _build(self):
        # Mağara göğü: ay yerine koyu kızıl tavan ve havada süzülen közler.
        self._mk(model='sphere',scale=500,
                 color=color.rgb32(8, 2, 3),double_sided=True, unlit=True, shader=None)
        scene.fog_color=self.atmosphere_fog_color
        scene.fog_density=self.atmosphere_fog_density

        # Lava zemini: parlak turuncu/kızıl taşlar; aydınlatma kaynakları shader'da hesaplanır.
        self.lava_hotspots = []
        self.lava_embers = []
        self.elapsed = 0.0
        for lx in range(-10,11):
            for lz in range(-5,22):
                br = random.randint(190, 248)
                green = random.randint(38, 88)
                blue = random.randint(0, 8)
                lava_tile = self._mk(model='cube',
                         position=(lx*2,-0.55,lz*2),
                         scale=(2.02,0.3,2.02),
                         color=color.rgb32(br, green, blue), shader=None if reflection_enabled else torch_lighting_shader)
                register_reflection_surface(lava_tile)
                mark_glow(lava_tile)

        # Hareketli sarı-turuncu magma çatlakları: kendi ışıklarını yayarlar.
        for _ in range(78):
            cx = random.uniform(-19.0, 19.0)
            cz = random.uniform(-8.0, 41.0)
            length = random.uniform(0.28, 1.35)
            crack = self._mk(model='cube', position=(cx, -0.385, cz),
                             scale=(length, 0.018, random.uniform(0.035, 0.10)),
                             rotation_y=random.uniform(0, 180),
                             color=color.rgb32(random.randint(255,255), random.randint(115,205), random.randint(2,15)),
                             unlit=True, shader=None, collider=None)
            self.lava_hotspots.append((crack, random.uniform(0.7, 1.5), random.uniform(0, math.tau)))

        # Birbirine binen sıcak ışık alanları bütün parkuru turuncuya boyar.
        lava_light_spots = [(-16, -4), (-7, 7), (0, 17), (8, 5), (16, 24), (2, 35), (-13, 31)]
        for lx, lz in lava_light_spots:
            register_warm_light_at(self, (lx, -0.14, lz), tint=(1.0, 0.20, 0.015),
                                   radius=13.0, intensity=2.4)
            add_warm_ground_glow(self, lx, lz, floor_y=-0.385, radius=5.6,
                                 tint=(255, 62, 5))

        # Lavdan yükselen közler; ortamın canlılığını artırır.
        for _ in range(34):
            ex = random.uniform(-18.0, 18.0)
            ez = random.uniform(-7.0, 40.0)
            ey = random.uniform(-0.1, 2.0)
            mote = self._mk(model='sphere', position=(ex, ey, ez),
                            scale=random.uniform(0.025, 0.075),
                            color=color.rgb32(255, random.randint(90, 185), 18),
                            unlit=True, shader=None, collider=None)
            self.lava_embers.append((mote, random.uniform(0.25, 0.8),
                                     random.uniform(0, math.tau), ey))

        # === PLATFORMLAR ===
        platforms=[
            ((0,   0.3,  0), (6,0.45,6)),
            ((7,   0.8,  5), (4,0.45,4)),
            ((-6,  1.3,  7), (4,0.45,4)),
            ((0,   1.8, 11), (4.5,0.45,4.5)),
            ((6,   2.3, 15), (4,0.45,4)),
            ((0,   2.8, 19), (5.5,0.45,5.5)),
        ]
        for pos,scl in platforms:
            # Platform taş rengi
            c_=color.rgb32(
                random.randint(65,88),
                random.randint(55,72),
                random.randint(55,72))
            self._mk(model='cube',position=pos,scale=scl,
                     color=c_,collider='box')
            # Platform kenar çizgisi
            self._mk(model='cube',
                     position=(pos[0],pos[1]+0.24,pos[2]),
                     scale=(scl[0],0.06,scl[2]),
                     color=color.rgb32(100,70,70))

        # Mağara duvarları (arka plan)
        for side,sx in [(-1,-16),(1,16)]:
            for wz in range(-2,22,3):
                h=random.uniform(6,12)
                self._mk(model='cube',
                         position=(sx,h/2,wz),
                         scale=(2,h,3),
                         color=color.rgb32(
                             random.randint(50,70),
                             random.randint(20,35),
                             random.randint(15,25)))

        # Stalaktitler (tavandan sarkan)
        for _ in range(18):
            px=random.uniform(-12,12)
            pz=random.uniform(-3,20)
            h=random.uniform(1.0,3.5)
            self._mk(model=cyl(),
                     scale=(0.25,h,0.25),
                     position=(px,9-h/2,pz),
                     color=color.rgb32(
                         random.randint(55,80),
                         random.randint(20,35),
                         random.randint(15,25)))

        # Ateş efektleri
        for fx,fz in [(-8,2),(8,2),(-8,12),(8,12),(0,6)]:
            for i in range(5):
                self._mk(model='sphere',
                         position=(fx,i*0.4-0.3,fz),
                         scale=0.38-i*0.04,
                         color=color.rgb32(255,max(20,120-i*25),0))
            register_warm_light_at(self, (fx, 1.5, fz), tint=(1.0, 0.28, 0.025),
                                   radius=10.5, intensity=1.9)
            add_warm_ground_glow(self, fx, fz, floor_y=-0.385, radius=2.8,
                                 tint=(255, 75, 8))

        # === PRENSESİN KAFESİ ===
        cp=Vec3(0,3.7,19)
        # Kafes çubukları
        for ang in range(0,360,40):
            rx=math.cos(math.radians(ang))*0.85
            rz=math.sin(math.radians(ang))*0.85
            self._mk(model=cyl(),scale=(0.07,2.4,0.07),
                     position=(cp.x+rx,cp.y,cp.z+rz),
                     color=color.rgb32(80,55,15))
        # Kafes taban & tavan
        for dy in [-1.2,1.2]:
            self._mk(model=cyl(),scale=(1.8,0.1,1.8),
                     position=(cp.x,cp.y+dy,cp.z),
                     color=color.rgb32(90,65,20))
        # Kafes ışığı
        register_warm_light_at(self, (cp.x, cp.y + 2, cp.z),
                               tint=(1.0, 0.45, 0.22), radius=9.0, intensity=1.15)

        # Prenses gövde
        self.princess=self._mk(
            model='cube',scale=(0.65,1.5,0.65),
            position=(cp.x,cp.y-0.45,cp.z),
            color=color.rgb32(230,170,195),collider='box')
        # Prenses kafa
        self._mk(model='sphere',scale=0.6,
                 position=(cp.x,cp.y+0.6,cp.z),
                 color=color.rgb32(240,200,170))
        # Saç
        self._mk(model='sphere',scale=0.65,
                 position=(cp.x,cp.y+0.65,cp.z-0.05),
                 color=color.rgb32(180,100,30))
        # Taç
        for ta in range(0,360,72):
            self._mk(model='cube',scale=(0.08,0.18,0.08),
                     position=(
                         cp.x+math.cos(math.radians(ta))*0.22,
                         cp.y+0.95,
                         cp.z+math.sin(math.radians(ta))*0.22),
                     color=color.rgb32(255,200,0))

        # Etiket
        lbl=self._mk(model='quad',scale=(2.9,0.48),
                     position=(cp.x,cp.y+1.95,cp.z),
                     billboard=True,
                     color=color.rgba32(20,8,30,220))
        Text('[ E ]  Prensesi Kurtar!',parent=lbl,
             scale=6.5,origin=(0,0),
             color=color.rgb32(255,170,255))

    def update(self):
        self.elapsed += time.dt
        for crack, speed, phase in getattr(self, 'lava_hotspots', []):
            pulse = 0.5 + 0.5 * math.sin(self.elapsed * speed * 2.5 + phase)
            crack.color = color.rgb32(255, int(100 + 115 * pulse), int(3 + 13 * pulse))
            crack.scale_y = 0.012 + 0.012 * pulse
        for mote, speed, phase, base_y in getattr(self, 'lava_embers', []):
            mote.y += time.dt * speed
            mote.x += math.sin(self.elapsed * 1.7 + phase) * time.dt * 0.22
            if mote.y > 3.8:
                mote.y = -0.25
                mote.x = random.uniform(-18.0, 18.0)
                mote.z = random.uniform(-7.0, 40.0)
        if not g.dlg_open and distance_xz(player.position,self.princess.position)<2.5:
            hud.set_hint('[ E ]  Prensesi Kurtar!')
            if held_keys['e']:
                hud.set_hint('')
                dlg.show(
                    "TEBRİKLER! \n\n"
                    "Prensesi kurtardın!\n"
                    "Karanlık Kral'ın büyüsü bozuldu.\n\n"
                    "Sen gerçek bir kahraman oldun!",
                    'OYUN TAMAMLANDI')
        else:
            hud.set_hint('')

    def clear(self):
        for e in self.ents:
            try: destroy(e)
            except: pass
        scene.fog_density=0


# ── PROFESYONEL KATEGORİLİ AYARLAR MENÜSÜ ───────────────────────
settings_open = False
graphics_quality = 'Yüksek'
advanced_lighting_enabled = True
saved_player_speed = 6
saved_jump_height = 2.4
menu_time = 0.0
last_settings_category = 'Genel Grafikler'

settings_root = Entity(parent=camera.ui, enabled=False, z=0, scale=(.94, .92))
Entity(parent=settings_root, model='quad', scale=(2,2), color=color.rgba32(2,5,14,238), z=0)
Entity(parent=settings_root, model='quad', scale=(1.86,1.48), color=color.rgba32(9,15,28,250), z=-0.01)
Entity(parent=settings_root, model='quad', scale=(1.82,1.44), color=color.rgba32(17,25,41,250), z=-0.015)
settings_glow = Entity(parent=settings_root, model='quad', scale=(1.90,1.56), color=color.rgba32(45,105,165,18), z=-0.02)
Entity(parent=settings_root, model='quad', scale=(1.82,0.012), y=0.70, color=color.rgb32(70,190,255), z=-0.025)
Text('KARANLIK KRAL', parent=settings_root, x=-0.82, y=0.61, scale=.98, origin=(-.5,0), color=color.rgb32(115,210,255), z=-0.04)
Text('SİSTEM AYARLARI', parent=settings_root, x=-0.82, y=0.51, scale=.65, origin=(-.5,0), color=color.rgb32(165,180,202), z=-0.04)
Entity(parent=settings_root, model='quad', x=-0.30, y=-0.01, scale=(0.004,1.18), color=color.rgb32(50,68,91), z=-0.03)

category_panels = {}
for _cat in ('Genel Grafikler','Işın İzleme','Kare Artırıcı'):
    category_panels[_cat] = Entity(parent=settings_root, enabled=False, z=-0.04)

def show_settings_category(name):
    global last_settings_category
    if name not in category_panels: return
    last_settings_category = name
    for cat, panel in category_panels.items(): panel.enabled = (cat == name)
    for cat, button in category_buttons.items():
        selected = cat == name
        button.color = color.rgb32(28,112,151) if selected else color.rgb32(25,35,53)
        button.highlight_color = color.rgb32(55,145,185) if selected else color.rgb32(55,72,96)

category_buttons = {}
def _category_button(label, y):
    b = Button(parent=settings_root, text=label, x=-.66, y=y, scale=(.50,.082),
               color=color.rgb32(25,35,53), highlight_color=color.rgb32(55,72,96),
               pressed_color=color.rgb32(40,130,165), text_color=color.white, z=-.06)
    b.on_click = lambda label=label: show_settings_category(label)
    b.ignore_paused = True
    # Give long category names enough room while keeping them readable.
    try:
        b.text_entity.scale = 0.93
    except Exception:
        pass
    category_buttons[label] = b
    return b
_category_button('Genel Grafikler', .34)
_category_button('Işın İzleme', .16)
_category_button('Kare Artırıcı', -.02)

# Genel Grafikler paneli
_general = category_panels['Genel Grafikler']
Text('GÖRSEL KALİTE', parent=_general, x=-.12, y=.38, scale=.78, origin=(-.5,0), color=color.rgb32(190,205,225), z=-.05)
quality_label = Text('SEÇİLİ: YÜKSEK', parent=_general, x=-.12, y=.28, scale=.72, origin=(-.5,0), color=color.rgb32(100,220,255), z=-.05)

def _make_panel_button(parent, label, x, y, action, scale=(.23,.064)):
    b = Button(parent=parent, text=label, x=x, y=y, scale=scale,
               color=color.rgb32(30,43,63), highlight_color=color.rgb32(58,119,155),
               pressed_color=color.rgb32(65,150,185), text_color=color.white, z=-.07)
    b.on_click = action
    b.ignore_paused = True
    try:
        b.text_entity.scale = 0.86
    except Exception:
        pass
    return b

quality_buttons = []
for label, x in [('DÜŞÜK',-.07),('ORTA',.20),('YÜKSEK',.47)]:
    quality_buttons.append(_make_panel_button(_general,label,x,.18,lambda level={'DÜŞÜK':'Düşük','ORTA':'Orta','YÜKSEK':'Yüksek'}[label]:apply_graphics_quality(level)))
Text('Oyun içi sis, gölge ve son işlem kalitesini yönetir.', parent=_general, x=-.12, y=-.01, scale=.50, origin=(-.5,0), color=color.rgb32(150,170,195), z=-.05)
Text('ÇIKIŞ ÇÖZÜNÜRLÜĞÜ', parent=_general, x=-.12, y=-.14, scale=.70, origin=(-.5,0), color=color.rgb32(190,205,225), z=-.05)
_make_panel_button(_general,'720p',.00,-.27,lambda:set_output_resolution('720p'),scale=(.19,.06))
_make_panel_button(_general,'900p',.28,-.27,lambda:set_output_resolution('900p'),scale=(.19,.06))
_make_panel_button(_general,'1080p',.56,-.27,lambda:set_output_resolution('1080p'),scale=(.19,.06))

# Işın İzleme paneli
_ray_panel = category_panels['Işın İzleme']
ray_label = Text('EKRAN-UZAYI IŞIN İZLEME', parent=_ray_panel, x=-.12, y=.38, scale=.70, origin=(-.5,0), color=color.rgb32(255,180,95), z=-.05)
ray_note = Text('SSRT: ekran uzayındaki yansımalar ve ortam gölgeleme.', parent=_ray_panel, x=-.12, y=.27, scale=.50, origin=(-.5,0), color=color.rgb32(155,175,198), z=-.05)
_make_panel_button(_ray_panel,'IŞIN İZLEME EFEKTLERİ: AÇ / KAPAT',.18,.15,lambda:apply_ray_tracing(not advanced_lighting_enabled),scale=(.56,.062))
config_bounce_label = Text('BOUNCE: 0', parent=_ray_panel, x=-.12, y=-.01, scale=.70, origin=(-.5,0), color=color.rgb32(255,190,100), z=-.05)
_make_panel_button(_ray_panel,'BOUNCE −',.04,-.13,lambda:set_bounce_count(rt_bounces-1),scale=(.24,.07))
_make_panel_button(_ray_panel,'BOUNCE +',.32,-.13,lambda:set_bounce_count(rt_bounces+1),scale=(.24,.07))
config_steps_label = Text('IŞIN ADIMI: 48', parent=_ray_panel, x=-.12, y=-.27, scale=.70, origin=(-.5,0), color=color.rgb32(255,190,100), z=-.05)
_make_panel_button(_ray_panel,'ADIM −',.04,-.37,lambda:set_trace_steps(rt_trace_steps-4),scale=(.24,.07))
_make_panel_button(_ray_panel,'ADIM +',.32,-.37,lambda:set_trace_steps(rt_trace_steps+4),scale=(.24,.07))

# Kare artırıcı paneli
_up_panel = category_panels['Kare Artırıcı']
upscaler_label = Text('KARE ARTTIRICI: AÇIK', parent=_up_panel, x=-.12, y=.38, scale=.70, origin=(-.5,0), color=color.rgb32(110,245,165), z=-.05)
upscaler_note = Text('Performans ve Dengeli profillerinde görüş dışı nesne önbelleği çalışır.', parent=_up_panel, x=-.12, y=.27, scale=.47, origin=(-.5,0), color=color.rgb32(155,175,198), z=-.05)
_make_panel_button(_up_panel,'KARE ARTTIRICI: AÇ / KAPAT',.18,.15,lambda:apply_frame_upscaler(not upscaler_enabled),scale=(.56,.062))
upscaler_profile_heading = Text('İÇ ÇÖZÜNÜRLÜK PROFİLİ', parent=_up_panel, x=-.12, y=.00, scale=.67, origin=(-.5,0), color=color.rgb32(190,205,225), z=-.05)
upscaler_profile_label = Text('PROFİL: YÜKSEK', parent=_up_panel, x=-.12, y=-.09, scale=.58, origin=(-.5,0), color=color.rgb32(100,220,255), z=-.05)
upscaler_profile_buttons=[]
for label, profile, x in [('PERFORMANS','Performans',-.02),('DENGELİ','Dengeli',.25),('YÜKSEK','Yüksek',.52)]:
    upscaler_profile_buttons.append(_make_panel_button(_up_panel,label,x,-.19,lambda profile=profile:apply_upscaler_profile(profile),scale=(.24,.07)))
Text('ARKA NESNE ÖNBELLEĞİ', parent=_up_panel, x=-.12, y=-.29, scale=.65, origin=(-.5,0), color=color.rgb32(190,205,225), z=-.05)
culling_delay_label = Text('Görüş dışında kalma süresi: 10 sn', parent=_up_panel, x=-.12, y=-.38, scale=.54, origin=(-.5,0), color=color.rgb32(100,220,255), z=-.05)

def set_culling_delay(seconds):
    global VIEW_UNLOAD_DELAY
    VIEW_UNLOAD_DELAY = max(1.0, min(30.0, float(seconds)))
    culling_delay_label.text = f'Görüş dışında kalma süresi: {VIEW_UNLOAD_DELAY:g} sn'

_make_panel_button(_up_panel,'− 1 sn',.04,-.48,lambda:set_culling_delay(VIEW_UNLOAD_DELAY-1),scale=(.22,.065))
_make_panel_button(_up_panel,'+ 1 sn',.31,-.48,lambda:set_culling_delay(VIEW_UNLOAD_DELAY+1),scale=(.22,.065))

# Alt sabit kontrol alanı
Entity(parent=settings_root, model='quad', x=.28, y=-.61, scale=(1.72,.003), color=color.rgb32(48,65,86), z=-.05)
_make_panel_button(settings_root,'OYUNA DÖN',.66,-.66,lambda: close_settings(),scale=(.30,.058))

# Animasyonlu ama düşük maliyetli menü parçacıkları
menu_sparks=[]
for i in range(18):
    spark=Entity(parent=settings_root, model='quad', scale=(random.uniform(.002,.005),random.uniform(.008,.02)),
                 x=random.uniform(-.7,.7), y=random.uniform(-.6,.6), color=color.rgba32(70,170,255,random.randint(25,75)), z=-.02)
    spark.ignore_paused=True
    menu_sparks.append((spark,random.uniform(.2,.8)))

OUTPUT_RESOLUTIONS={'720p':(1280,720),'900p':(1600,900),'1080p':(1920,1080)}
current_output_resolution='1080p'
def set_output_resolution(name):
    global current_output_resolution, UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT
    if name not in OUTPUT_RESOLUTIONS: return
    current_output_resolution=name
    UPSCALE_OUTPUT_WIDTH,UPSCALE_OUTPUT_HEIGHT=OUTPUT_RESOLUTIONS[name]
    try:
        # Seçilen çıkış çözünürlüğünü pencereye uygula; pencere modunda ekran boyutunu
        # yeni çözünürlüğe uydur, render hedeflerini de yeniden boyutlandır.
        window.fullscreen = False
        window.size = (UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT)
        try:
            from panda3d.core import WindowProperties
            props = WindowProperties()
            props.setSize(int(UPSCALE_OUTPUT_WIDTH), int(UPSCALE_OUTPUT_HEIGHT))
            base.win.requestProperties(props)
        except Exception:
            pass
    except Exception as exc:
        print('[CONFIG] Çözünürlük uygulanamadı:', exc)
    try:
        if RT_PIPELINE_AVAILABLE and rt_manager is not None: rt_manager.resizeBuffers()
        update_raytrace_uniforms()
    except Exception as exc: print('[CONFIG] Render hedefi güncellenemedi:',exc)

def set_bounce_count(value):
    global rt_bounces
    rt_bounces=max(0,min(4,int(value)))
    if RT_PIPELINE_AVAILABLE and rt_quad is not None:
        try: rt_quad.setShaderInput('maxBounces',int(rt_bounces))
        except Exception as exc: print('[CONFIG] Bounce uygulanamadı:',exc)
    if 'config_bounce_label' in globals(): config_bounce_label.text=f'BOUNCE: {rt_bounces}'

def set_trace_steps(value):
    global rt_trace_steps
    rt_trace_steps=max(4,min(96,int(value)))
    if RT_PIPELINE_AVAILABLE and rt_quad is not None:
        try: rt_quad.setShaderInput('traceSteps',int(rt_trace_steps))
        except Exception as exc: print('[CONFIG] Işın adımı uygulanamadı:',exc)
    if 'config_steps_label' in globals(): config_steps_label.text=f'IŞIN ADIMI: {rt_trace_steps}'

def apply_graphics_quality(level):
    global graphics_quality
    graphics_quality=level
    quality_label.text='SEÇİLİ: '+level.upper()
    active_chapter=globals().get('current_ch')
    fog_color=getattr(active_chapter,'atmosphere_fog_color',color.rgb32(7,10,24))
    fog_density=getattr(active_chapter,'atmosphere_fog_density',.026)
    if level=='Düşük': scene.fog_density=fog_density*.72; _set_shadow_quality(512)
    elif level=='Orta': scene.fog_density=fog_density; _set_shadow_quality(1024)
    else: scene.fog_density=fog_density*1.12; _set_shadow_quality(3072)
    scene.fog_color=fog_color
    set_postfx_quality(level)
    active=advanced_lighting_enabled and level!='Düşük' and (RT_PIPELINE_AVAILABLE or reflection_enabled)
    set_reflection_pass(active)
    if RT_PIPELINE_AVAILABLE:
        ray_label.text='EKRAN-UZAYI IŞIN İZLEME: '+('AÇIK' if active else 'KAPALI')
        ray_label.color=color.rgb32(110,245,165) if active else color.rgb32(255,180,95)
    else:
        ray_label.text='GPU RT SHADER BAŞLATILAMADI'; ray_label.color=color.rgb32(255,90,90)
    for b in quality_buttons:
        selected=b.text.upper()==level.upper(); b.color=color.rgb32(35,115,160) if selected else color.rgb32(35,43,62)

def apply_ray_tracing(enabled):
    global advanced_lighting_enabled
    advanced_lighting_enabled=bool(enabled)
    active=advanced_lighting_enabled and graphics_quality!='Düşük' and (RT_PIPELINE_AVAILABLE or reflection_enabled)
    ray_label.text='EKRAN-UZAYI IŞIN İZLEME: '+('AÇIK' if active else 'KAPALI') if RT_PIPELINE_AVAILABLE else 'GPU RT SHADER BAŞLATILAMADI'
    ray_label.color=color.rgb32(110,245,165) if active else color.rgb32(255,180,95)
    set_reflection_pass(active)
    _set_shadow_quality({'Düşük':768,'Orta':1536,'Yüksek':3072}[graphics_quality])
    set_postfx_quality(graphics_quality)

def apply_frame_upscaler(enabled):
    global upscaler_enabled
    requested=bool(enabled)
    if not RT_PIPELINE_AVAILABLE or rt_manager is None or rt_quad is None:
        upscaler_enabled=False; upscaler_label.text='KARE ARTTIRICI: KULLANILAMIYOR'; upscaler_label.color=color.rgb32(255,105,105)
        upscaler_note.text='Render hattı başlatılamadığı için ölçekleme uygulanamadı.'; return
    upscaler_enabled=requested
    try:
        rt_manager.forcex=UPSCALE_SOURCE_WIDTH if requested else 0
        rt_manager.forcey=UPSCALE_SOURCE_HEIGHT if requested else 0
        rt_manager.resizeBuffers()
        rt_color_tex.setMinfilter(Texture.FTLinear); rt_color_tex.setMagfilter(Texture.FTLinear)
        rt_depth_tex.setMinfilter(Texture.FTNearest); rt_depth_tex.setMagfilter(Texture.FTNearest)
        rt_quad.setShaderInput('upscalerEnabled',1 if requested else 0)
        rt_quad.setShaderInput('upscaleSharpness',.42 if requested else 0.0)
        upscaler_label.text='KARE ARTTIRICI: '+('AÇIK' if requested else 'KAPALI')
        upscaler_label.color=color.rgb32(110,245,165) if requested else color.rgb32(255,180,95)
        upscaler_note.text='Performans/Dengeli: görüş dışı nesne önbelleği açık.' if requested else 'Kapalı: nesne önbelleği otomatik devre dışı.'
        update_raytrace_uniforms()
    except Exception as exc:
        upscaler_enabled=False; upscaler_label.text='KARE ARTTIRICI: HATA'; upscaler_note.text=str(exc)[:90]
        print('[KARE ARTTIRICI] Ölçekleme uygulanamadı:',exc)

def apply_upscaler_profile(profile, automatic=False):
    global upscaler_profile, UPSCALE_SOURCE_WIDTH, UPSCALE_SOURCE_HEIGHT, UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT
    global auto_perf_active, auto_perf_has_triggered, auto_saved_graphics_quality
    if profile not in UPSCALE_PROFILES: return
    if not automatic:
        # Kullanıcı elle profil seçtiyse adaptif modun yaptığı grafik düşürmesini geri al.
        if auto_perf_active and profile != 'Performans' and auto_saved_graphics_quality in ('Düşük','Orta','Yüksek'):
            auto_perf_active=False
            try:
                if globals().get('graphics_quality') != auto_saved_graphics_quality:
                    apply_graphics_quality(auto_saved_graphics_quality)
            except Exception:
                pass
        else:
            auto_perf_active=False
        auto_perf_has_triggered = (profile == 'Performans')
    else:
        auto_perf_active=True
        auto_perf_has_triggered=True
    upscaler_profile=profile
    source_w,source_h,_ow,_oh=UPSCALE_PROFILES[profile]
    UPSCALE_SOURCE_WIDTH,UPSCALE_SOURCE_HEIGHT=source_w,source_h
    output_size=OUTPUT_RESOLUTIONS.get(current_output_resolution,(1920,1080))
    UPSCALE_OUTPUT_WIDTH,UPSCALE_OUTPUT_HEIGHT=output_size
    try: window.size=(UPSCALE_OUTPUT_WIDTH,UPSCALE_OUTPUT_HEIGHT)
    except Exception as exc: print('[KARE ARTTIRICI] Çıkış çözünürlüğü değiştirilemedi:',exc)
    try:
        if RT_PIPELINE_AVAILABLE and rt_manager is not None:
            rt_manager.forcex=UPSCALE_SOURCE_WIDTH if upscaler_enabled else 0
            rt_manager.forcey=UPSCALE_SOURCE_HEIGHT if upscaler_enabled else 0
            rt_manager.resizeBuffers()
        apply_frame_upscaler(upscaler_enabled)
    except Exception as exc: print('[KARE ARTTIRICI] Profil uygulanamadı:',exc)
    upscaler_profile_label.text=f'PROFİL: {profile.upper()}  |  {source_w}x{source_h} to {UPSCALE_OUTPUT_WIDTH}x{UPSCALE_OUTPUT_HEIGHT}'
    for b in upscaler_profile_buttons:
        selected=b.text.upper()==profile.upper(); b.color=color.rgb32(35,125,155) if selected else color.rgb32(35,43,62)
    update_raytrace_uniforms()

def _set_shadow_quality(resolution):
    try:
        if resolution<=0: sun._light.setShadowCaster(False); sun.shadows=False
        else:
            sun.shadow_map_resolution=Vec2(resolution,resolution)
            sun._light.setShadowCaster(True,int(resolution),int(resolution)); sun.shadows=True; sun.update_bounds(shadow_bounds)
    except Exception as exc: print('[GÖLGE] Kalite değiştirilemedi:',exc)

def open_settings():
    global settings_open
    if settings_open: return
    settings_open=True; settings_root.enabled=True
    if not hasattr(open_settings,'_saved'): open_settings._saved=(player.speed,player.jump_height)
    player.speed=0; player.jump_height=0; mouse.locked=False; mouse.visible=True
    application.paused=True
    show_settings_category(last_settings_category)

def close_settings():
    global settings_open
    settings_open=False; settings_root.enabled=False
    if hasattr(open_settings,'_saved'):
        player.speed,player.jump_height=open_settings._saved; del open_settings._saved
    application.paused=False; mouse.locked=True; mouse.visible=False

for _menu_entity in settings_root.children:
    _menu_entity.ignore_paused=True
for _cat_panel in category_panels.values():
    for _ent in _cat_panel.children: _ent.ignore_paused=True
show_settings_category(last_settings_category)

# ─────────────────────────────────────────────────────────────
#  OYUNCU & BÖLÜM YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────
player=FirstPersonController(speed=6,jump_height=2.4,gravity=0.9,
                              mouse_sensitivity=Vec2(40,40))
mouse.locked=True

# Subtle atmospheric motes: tiny emissive particles that drift around the camera.
ambient_motes = []
for _ in range(36):
    mote = Entity(model='sphere', scale=random.uniform(0.018, 0.045),
                  position=(random.uniform(-12, 12), random.uniform(0.4, 5.5), random.uniform(-12, 12)),
                  color=color.rgba32(80, 155, 255, random.randint(90, 180)),
                  unlit=True, collider=None)
    mark_glow(mote)
    ambient_motes.append((mote, random.uniform(0.08, 0.28), random.uniform(0.0, math.tau)))

current_ch=None

# Adaptif performans yöneticisi. FPS 25'in altında istikrarlı biçimde kalırsa
# tek seferlik Performans profiline geçer; ayar elle değiştirilince yeniden hazır olur.
AUTO_PERFORMANCE_ENABLED = True
auto_perf_active = False
auto_perf_has_triggered = False
auto_perf_low_time = 0.0
auto_perf_warmup = 0.0
auto_perf_fps_smooth = 60.0
auto_saved_graphics_quality = None

# Görüş alanı / mesafe / duvar arkası elemesi. Yalnızca aktif bölümdeki modeller taranır.
# Entity'ler ve collider'lar silinmez; önbellekten geri açılabilirler.
view_culling_enabled = True
view_culling_interval = 0.14       # CPU giderini azaltmak için yaklaşık 7 kez/sn
view_culling_timer = 0.0
occlusion_cache = {}               # id(entity) -> (duvar arkası mı, son kontrol zamanı)
occlusion_cursor = 0               # pahalı ray testlerini bölümlere yay
OCCLUSION_RAY_BUDGET = 10          # her görüş taramasında en fazla 10 ray testi
# id(entity) -> (entity, önceki visible değeri)
view_culling_hidden = {}
# 10 saniyeden uzun süre görüş dışında kalanlar render dışına alınır;
# Entity referansı ve önceki enabled durumu bellekte tutulur; model nesnesi korunur.
view_culling_unloaded = {}  # id(entity) -> (entity, was_enabled)
VIEW_UNLOAD_DELAY = 10.0


def _is_alive_entity(entity):
    try:
        return entity is not None and not entity.is_empty()
    except Exception:
        return entity is not None


def restore_view_culled_entities():
    """Gizlenen/önbelleğe alınan modelleri önceki görünürlükleriyle geri yükle."""
    global view_culling_hidden, view_culling_unloaded
    for eid, (entity, was_enabled) in list(view_culling_unloaded.items()):
        if _is_alive_entity(entity):
            try:
                entity.enabled = was_enabled
            except Exception:
                pass
    view_culling_unloaded.clear()
    occlusion_cache.clear()
    for entity, was_visible in list(view_culling_hidden.values()):
        if _is_alive_entity(entity):
            try:
                entity.visible = was_visible
            except Exception:
                pass
    view_culling_hidden.clear()


def update_view_culling(force=False):
    """Görüş konisi + mesafe/LOD + sınırlı duvar arkası testi."""
    global view_culling_timer, occlusion_cursor
    # Görüş/LOD optimizasyonu ölçekleyiciden bağımsızdır; düşük profillerde devam eder.
    low_mid_profile = (upscaler_profile in ('Performans', 'Dengeli')) or auto_perf_active
    if not view_culling_enabled or not low_mid_profile or settings_open or current_ch is None:
        if view_culling_hidden or view_culling_unloaded:
            restore_view_culled_entities()
        return

    view_culling_timer += time.dt
    if not force and view_culling_timer < view_culling_interval:
        return
    view_culling_timer = 0.0

    try:
        forward = Vec3(camera.forward.x, 0, camera.forward.z)
        if forward.length() < 0.0001:
            forward = Vec3(player.forward.x, 0, player.forward.z)
        if forward.length() < 0.0001:
            return
        forward = forward.normalized()
        cam_pos = Vec3(camera.world_position.x, 0, camera.world_position.z)

        # Histerezis ve profil bazlı görüş uzaklığı.
        if auto_perf_active or upscaler_profile == 'Performans':
            max_render_distance, restore_distance = 40.0, 46.0
            lod_hide_distance, lod_restore_distance = 23.0, 19.0
        elif upscaler_profile == 'Dengeli':
            max_render_distance, restore_distance = 54.0, 60.0
            lod_hide_distance, lod_restore_distance = 31.0, 27.0
        else:
            max_render_distance, restore_distance = 72.0, 78.0
            lod_hide_distance, lod_restore_distance = 43.0, 39.0
        near_keep_distance = 5.5
        hide_dot_limit = 0.02
        restore_dot_limit = -0.10
        now_mono = _time.monotonic()
        ray_budget = OCCLUSION_RAY_BUDGET
        entities = list(getattr(current_ch, 'ents', []))
        if entities:
            start_idx = occlusion_cursor % len(entities)
            entities = entities[start_idx:] + entities[:start_idx]

        for entity in entities:
            if not _is_alive_entity(entity):
                view_culling_hidden.pop(id(entity), None)
                continue

            eid = id(entity)
            try:
                if getattr(entity, '_kk_no_view_cull', False):
                    if eid in view_culling_hidden:
                        entity.visible = view_culling_hidden[eid][1]
                        view_culling_hidden.pop(eid, None)
                    continue

                scale = entity.scale
                try:
                    sx = abs(float(scale.x)); sy = abs(float(scale.y)); sz = abs(float(scale.z))
                    # Yaklaşık yarıçap: iri ağaç/kaya görüş konisine daha erken girer.
                    radius = max(sx, sy, sz) * 0.5
                    largest_scale = max(sx, sy, sz)
                except Exception:
                    largest_scale = abs(float(scale)) if isinstance(scale, (int, float)) else 1.0
                    radius = largest_scale * 0.5

                # Gökyüzü kubbesi ve çok büyük çevre kabukları sabit kalmalı.
                if largest_scale > 100:
                    if eid in view_culling_hidden:
                        entity.visible = view_culling_hidden[eid][1]
                        view_culling_hidden.pop(eid, None)
                    continue

                pos = entity.world_position
                delta = Vec3(pos.x - cam_pos.x, 0, pos.z - cam_pos.z)
                distance = delta.length()
                hidden_entry = view_culling_hidden.get(eid)
                unloaded_entry = view_culling_unloaded.get(eid)

                if distance <= near_keep_distance:
                    should_render = True
                elif hidden_entry is not None:
                    # Gizli nesneler için daha geniş geri-getirme sınırı kullan.
                    direction = delta / max(distance, 0.0001)
                    dot = forward.dot(direction)
                    should_render = (distance <= restore_distance + radius and
                                     dot >= restore_dot_limit - min(radius / max(distance, 1.0), 0.25))
                elif distance > max_render_distance + radius:
                    should_render = False
                else:
                    direction = delta / max(distance, 0.0001)
                    dot = forward.dot(direction)
                    should_render = dot >= hide_dot_limit - min(radius / max(distance, 1.0), 0.35)

                # LOD: uzaktaki küçük dekorları kaldırarak çizim çağrılarını azalt.
                # Şifre kâğıtları, etkileşimli ve collider taşıyan nesneler bu sınıfa girmez.
                if should_render and getattr(entity, '_kk_lod_candidate', False):
                    if hidden_entry is None:
                        should_render = distance <= lod_hide_distance
                    else:
                        should_render = distance <= lod_restore_distance

                # Duvar arkası elemesi: sadece görünür konideki küçük, collider'sız adaylarda
                # sınırlı sayıda raycast yapılır; sonuç kısa süre cache'lenerek FPS korunur.
                # Çarpışma duvarları/zeminleri kendileri test edilmez, sadece engel olarak kullanılır.
                if should_render and getattr(entity, '_kk_occlusion_candidate', False) and distance > 7.0 and distance < max_render_distance:
                    cache_entry = occlusion_cache.get(eid)
                    is_stale = (cache_entry is None or now_mono - cache_entry[1] > 0.55)
                    if is_stale and ray_budget > 0:
                        ray_budget -= 1
                        try:
                            origin = camera.world_position
                            raw_target = entity.world_position
                            # Zemine yakın küçük dekorlarda ışının zemine erken çarpıp
                            # nesneyi yanlışlıkla duvar arkasında sanmasını önlemek için hedefi biraz yükselt.
                            target_raise = max(0.0, min(1.0, (origin.y - raw_target.y) * 0.55))
                            target = raw_target + Vec3(0, target_raise, 0)
                            ray_vec = target - origin
                            ray_len = ray_vec.length() - max(0.35, radius * 0.35)
                            occluded = False
                            if ray_len > 0.25:
                                hit = raycast(origin, ray_vec.normalized(), distance=ray_len,
                                              ignore=(player,), debug=False)
                                occluded = bool(getattr(hit, 'hit', False))
                            occlusion_cache[eid] = (occluded, now_mono)
                            cache_entry = occlusion_cache[eid]
                        except Exception:
                            # GPU/driver/collider edge case: fail open so geometry is not lost.
                            occlusion_cache[eid] = (False, now_mono)
                            cache_entry = occlusion_cache[eid]
                    if cache_entry is not None and cache_entry[0] and now_mono - cache_entry[1] < 0.8:
                        should_render = False

                if should_render:
                    # Kamera geri döndü: model referansını önbellekten sahneye yeniden bağla.
                    if unloaded_entry is not None:
                        cached_entity, was_enabled = unloaded_entry
                        if _is_alive_entity(cached_entity):
                            # Modeli yeniden oluşturmak yerine aynı Entity'yi tekrar etkinleştir.
                            cached_entity.enabled = was_enabled
                        view_culling_unloaded.pop(eid, None)
                    if hidden_entry is not None:
                        entity.visible = hidden_entry[1]
                        try:
                            entity._kk_hidden_since = None
                        except Exception:
                            pass
                        view_culling_hidden.pop(eid, None)
                else:
                    # Önce görünürlüğü kapat; 10 saniye sonra Entity'yi devre dışı bırak.
                    # Modeli/entity'yi silme veya model alanını None yapma: dönüşte aynısı açılır.
                    if hidden_entry is None and bool(getattr(entity, 'visible', True)):
                        previous_visible = bool(entity.visible)
                        entity.visible = False
                        view_culling_hidden[eid] = (entity, previous_visible)
                        try:
                            entity._kk_hidden_since = _time.monotonic()
                        except Exception:
                            pass
                    if hidden_entry is not None and eid not in view_culling_unloaded:
                        hidden_entity, _was_visible = hidden_entry
                        hidden_since = getattr(hidden_entity, '_kk_hidden_since', None)
                        if hidden_since is None:
                            try:
                                hidden_entity._kk_hidden_since = _time.monotonic()
                                hidden_since = hidden_entity._kk_hidden_since
                            except Exception:
                                hidden_since = _time.monotonic()
                        if _time.monotonic() - hidden_since >= VIEW_UNLOAD_DELAY:
                            try:
                                was_enabled = bool(getattr(hidden_entity, 'enabled', True))
                                view_culling_unloaded[eid] = (hidden_entity, was_enabled)
                                hidden_entity.enabled = False
                            except Exception:
                                pass
            except Exception:
                # Tek bir özel Entity hata verirse kalan modellerin kontrolü devam etsin.
                continue
        if entities:
            # Her kontrolde başlangıç noktasını değiştir: tüm nesneler sıra ile ray test alabilsin.
            occlusion_cursor = (occlusion_cursor + max(1, len(entities) // 3)) % len(entities)
    except Exception as cull_error:
        print('[GÖRÜŞ CULLING] Güncelleme atlandı:', cull_error)


# ── Eldeki kılıç ──────────────────────────────────────────
hand_sword = None
HAND_ROT = Vec3(10, 0, -25)

def give_sword():
    global hand_sword
    if hand_sword: return
    hand_sword = Entity(parent=camera, position=(0.55,-0.55,1.2), rotation=HAND_ROT)
    Entity(parent=hand_sword, model='cube', scale=(0.09,0.95,0.04), y=0.5,
           color=color.rgb32(200,205,220))
    Entity(parent=hand_sword, model='cube', scale=(0.32,0.07,0.08), y=0.02,
           color=color.rgb32(210,170,60))
    Entity(parent=hand_sword, model='cube', scale=(0.07,0.26,0.07), y=-0.15,
           color=color.rgb32(80,50,25))

def swing():
    if not hand_sword: return
    hand_sword.animate('rotation', Vec3(75,0,-5), duration=0.1)
    invoke(lambda: hand_sword.animate('rotation', HAND_ROT, duration=0.2), delay=0.11)

def player_attack():
    if not g.has_sword or g.atk_cd > 0: return
    g.atk_cd = 0.4
    swing()
    if current_ch and hasattr(current_ch, 'on_player_attack'):
        current_ch.on_player_attack()

def hurt_player(dmg):
    g.hp = max(0, g.hp - dmg)
    hud.refresh()
    hud.hurt()
    if g.hp <= 0:
        g.respawn = True
        dlg.show("Karanlık Şövalye seni yere serdi...\n\nToparlan ve tekrar dene.", 'Yenildin')

def load_chapter(n):
    global current_ch, view_culling_timer
    # Önce eski bölümde görüş culling'i yüzünden gizlenen modelleri geri yükle, sonra bölümü temizle.
    restore_view_culled_entities()
    view_culling_timer = 0.0
    g.chapter=n
    if current_ch:
        current_ch.clear(); destroy(current_ch)
    player.position=Vec3(0,2,0)
    player.rotation=Vec3(0,0,0)
    if   n==1: current_ch=Chapter1()
    elif n==2: current_ch=Chapter2()
    elif n==3: current_ch=Chapter3()
    elif n==4: current_ch=Chapter4()
    # Şifre notları sadece şato keşif bölümünde görünür.
    if 'hud' in globals():
        hud.code_note.enabled = (n == 2)
        hud.code_note_bg.enabled = (n == 2)
        if n != 2:
            hud.code_note.text = ''
    # Ambient particles change palette with the location: forest, castle, boss arena, lava cave.
    palette = {1: color.rgb32(120, 180, 255),
               2: color.rgb32(255, 190, 100),
               3: color.rgb32(255, 145, 58),
               4: color.rgb32(255, 85, 28)}[n]
    for mote, _speed, _phase in ambient_motes:
        mote.color = palette
        mote.position = (player.x + random.uniform(-12, 12),
                         player.y + random.uniform(-1, 5),
                         player.z + random.uniform(-12, 12))
    hud.set_boss(None)
    hud.refresh()
    try:
        sun.update_bounds(shadow_bounds)
        update_warm_light_uniforms()
        apply_graphics_quality(graphics_quality)
    except Exception as visual_setup_error:
        print('[GÖRSEL] Bölüm ayarları uygulanamadı:', visual_setup_error)

def restart_run_after_fall():
    """Dünyanın altına düşülürse bütün ilerlemeyi temizleyip ormandan yeniden başlat."""
    global hand_sword, current_ch, view_culling_timer
    print('[OYUN] Dünya sınırının altına düşüldü; yeni oyun başlatılıyor.')
    # Açık şifre paneli/diyalog ve eski respawn durumu kapatılır.
    try:
        dlg.enabled = False
        dlg._show_id += 1
    except Exception:
        pass
    g.chapter = 1
    g.hp = 100
    g.has_sword = False
    g.has_key = False
    g.dlg_open = False
    g.enemy_dead = False
    g.atk_cd = 0.0
    g.respawn = False
    if hand_sword is not None:
        try: destroy(hand_sword)
        except Exception: pass
        hand_sword = None
    try:
        hud.code_note.text = ''
        hud.code_note.enabled = False
        hud.code_note_bg.enabled = False
        hud.set_boss(None)
        hud.set_hint('')
        hud.refresh()
    except Exception:
        pass
    restore_view_culled_entities()
    view_culling_timer = 0.0
    load_chapter(1)
    player.position = Vec3(0, 2.5, 0)
    player.rotation = Vec3(0, 0, 0)
    try:
        player.air_time = 0
        player.grounded = True
        player.jumping = False
        player.velocity = Vec3(0, 0, 0)
    except Exception:
        pass
    hud.announce('Dünya sınırının dışına düştün. Macera baştan başladı!', 3.5)

def update_auto_performance():
    """FPS 25 altına 1.6 sn düşerse Performans profili + düşük grafik kalitesi uygula."""
    global auto_perf_low_time, auto_perf_warmup, auto_perf_fps_smooth
    global auto_perf_active, auto_perf_has_triggered, auto_saved_graphics_quality
    if not AUTO_PERFORMANCE_ENABLED or settings_open:
        return
    dt = max(float(time.dt), 0.0001)
    instant_fps = min(300.0, 1.0 / dt)
    # Exponential smoothing avoids switching due to one short loading spike.
    alpha = min(1.0, dt * 1.8)
    auto_perf_fps_smooth += (instant_fps - auto_perf_fps_smooth) * alpha
    auto_perf_warmup += dt
    if auto_perf_warmup < 3.0 or auto_perf_has_triggered:
        return
    if auto_perf_fps_smooth < 25.0:
        auto_perf_low_time += dt
    else:
        auto_perf_low_time = max(0.0, auto_perf_low_time - dt * 0.75)
    if auto_perf_low_time < 1.6:
        return

    auto_perf_has_triggered = True
    auto_perf_active = True
    auto_saved_graphics_quality = globals().get('graphics_quality', 'Yüksek')
    try:
        if upscaler_profile != 'Performans':
            apply_upscaler_profile('Performans', automatic=True)
        else:
            auto_perf_active = True
        if globals().get('graphics_quality') != 'Düşük':
            apply_graphics_quality('Düşük')
        # Culling/LOD remains useful even if the optional upscale shader is unavailable.
        update_view_culling(force=True)
        msg = 'FPS 25 altına düştü: otomatik Performans profili etkin.'
        if not RT_PIPELINE_AVAILABLE:
            msg = 'FPS düşük: Performans LOD ve grafik optimizasyonları etkin. Ölçekleme hattı bu sistemde kullanılamıyor.'
        try:
            hud.announce(msg, 4.5)
        except Exception:
            pass
        print('[ADAPTİF FPS] Performans profiline geçildi. Ortalama FPS:', round(auto_perf_fps_smooth, 1))
    except Exception as auto_error:
        print('[ADAPTİF FPS] Otomatik profil uygulanamadı:', auto_error)


def update():
    global menu_time
    # Harita altına düşmek her bölümde tam yeniden başlatır: boss, anahtar, kılıç
    # ve şifre notları sıfırlanır; yeni koşu her zaman ormanda başlar.
    if 'player' in globals() and player.y < -14:
        restart_run_after_fall()
        return
    # Ek güvenlik: ayarlar açıkken oyun tarafındaki global güncelleme de tamamen durur.
    if settings_open:
        return
    update_auto_performance()
    # Bölüm Entity'lerinin update()'i Ursina tarafından zaten çağrılır.
    g.atk_cd = max(0.0, g.atk_cd - time.dt)
    update_warm_light_uniforms()
    update_reflection_camera()
    update_raytrace_uniforms()
    update_view_culling()
    particle_limit = 10 if (auto_perf_active or upscaler_profile == 'Performans') else (20 if upscaler_profile == 'Dengeli' else len(ambient_motes))
    for mote_index, (mote, speed, phase) in enumerate(ambient_motes):
        # Arka plan közleri performans profillerinde çizilmez; tekrar kalite yükselince geri açılır.
        should_enable_mote = mote_index < particle_limit
        if mote.enabled != should_enable_mote:
            mote.enabled = should_enable_mote
        if not should_enable_mote:
            continue
        mote.y += time.dt * speed
        mote.x += math.sin(_time.monotonic() * 0.7 + phase) * time.dt * 0.12
        mote.scale = 0.018 + (0.012 * (0.5 + 0.5 * math.sin(_time.monotonic() * 1.8 + phase)))
        if mote.y > player.y + 7:
            mote.y = player.y - random.uniform(1.0, 3.0)
        if abs(mote.x - player.x) > 18 or abs(mote.z - player.z) > 18:
            mote.position = (player.x + random.uniform(-12, 12),
                             player.y + random.uniform(-1, 5),
                             player.z + random.uniform(-12, 12))
    if settings_open:
        menu_time += time.dt
        player.speed = 0
        player.jump_height = 0
        settings_glow.alpha = 0.12 + (math.sin(menu_time * 2.1) + 1) * 0.045
        for spark, speed in menu_sparks:
            spark.y += time.dt * speed * 0.07
            if spark.y > 0.57:
                spark.y = -0.57
                spark.x = random.uniform(-0.82, 0.82)

def settings_input(key):
    # Bu handler ignore_paused=True: ESC ile menü hem açılır hem de kapatılır.
    if key == 'escape':
        if settings_open:
            close_settings()
        else:
            open_settings()

# Ayrı handler gerekir; global input duraklatılmışken çağrılmayabilir.
settings_pause_handler = Entity(ignore_paused=True, input=settings_input)

def input(key):
    # Şato şifre paneli önce girdileri yakalar; sonra normal oyun girdileri devam eder.
    if not settings_open and current_ch and hasattr(current_ch, 'handle_input'):
        if current_ch.handle_input(key):
            return
    # Enter yalnızca ekrandaki mevcut altyazıyı atlar.
    if key in ('enter', 'numpad enter') and dlg.enabled and not settings_open:
        dlg.hide()
        return
    if key == 'e' and not settings_open and not g.dlg_open and current_ch and hasattr(current_ch, 'interact'):
        current_ch.interact()
        return
    if key == 'left mouse down' and not settings_open:
        player_attack()

apply_graphics_quality('Yüksek')
apply_ray_tracing(True)
apply_upscaler_profile('Yüksek')
mouse.visible = False
load_chapter(1)
app.run()
