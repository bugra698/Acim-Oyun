from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
from ursina.shader import Shader as UrsinaShader
from panda3d.core import BitMask32, TextureStage, Vec4, Mat4, Texture, Shader, Vec2
import random, math

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
        rt_quad.setShaderInput('framePhase', float(time.time()))
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
    now = time.time()
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
shadow_bounds = Entity(model='cube', position=(0, 7, 16), scale=(36, 22, 64),
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
        self.ch_txt.text = f'Bölüm {g.chapter} / 3'
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
        e=Entity(*a,**kw); self.ents.append(e); return e

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
            flame.y = base_y + pulse * 0.035
            inner.y = flame.y + 0.06
            flame.scale = 0.27 + 0.025 * (0.5 + 0.5 * pulse)
            light['intensity'] = 1.45 + 0.45 * (0.5 + 0.5 * pulse)
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
#  BÖLÜM 2 – KALE
# ─────────────────────────────────────────────────────────────
class Chapter2(Entity):
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
            "Kale içindesin. Soğuk ve karanlık.\n\n"
            "Anahtarı bul, Karanlık Şövalye'yi yen,\n"
            "kuzey kapısından çık!\n\n"
            "Şövalye seni görünce üstüne gelir.\n"
            "Yakınındayken SOL TIK ile kılıcını salla.\n"
            "Darbesini hazırlarken geri çekilirsen kaçarsın!",
            'Bölüm 2 – Karanlık Kale'), delay=0.6)

    def _mk(self,*a,**kw):
        # Dünya modellerinde gölge alabilen ışıklandırma shader'ı.
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e=Entity(*a,**kw); self.ents.append(e); return e

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
            self._make_torch(tx, tz, base_y=5.78, pole_h=0.92)

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
            "Karanlık Şövalye yere yıkıldı!\nKale titredi.\n\nŞimdi kuzey kapısını aç!",
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
            flame.y = base_y + pulse * 0.035
            inner.y = flame.y + 0.06
            flame.scale = 0.27 + 0.025 * (0.5 + 0.5 * pulse)
            light['intensity'] = 1.55 + 0.50 * (0.5 + 0.5 * pulse)
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
                    dlg.show("Altın anahtar!\nBoss'u yenip kuzey\nkapısını açabilirsin.", 'Anahtar Bulundu')
                    return
        # Boss yakınında mı?
        if self.boss_alive and distance_xz(player.position, self.boss.position) < 3.6:
            hint = '[ SOL TIK ]  Saldır!'
        # Kapı
        if distance_xz(player.position, Vec3(0,2.5,-11.4)) < 3:
            if g.has_key and not self.boss_alive:
                hint = '[ E ]  Kapıyı Aç'
                if held_keys['e']: hud.set_hint(''); load_chapter(3); return
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
#  BÖLÜM 3 – EJDERHA YUVASI
# ─────────────────────────────────────────────────────────────
class Chapter3(Entity):
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
            'Bölüm 3 – Ejderha Yuvası'), delay=0.6)

    def _mk(self,*a,**kw):
        # Dünya modellerinde gölge alabilen ışıklandırma shader'ı.
        if kw.get('model') != 'sphere' or kw.get('scale') != 500:
            kw.setdefault('shader', torch_lighting_shader)
        e=Entity(*a,**kw); self.ents.append(e); return e

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


# ── GELİŞMİŞ AYARLAR MENÜSÜ ────────────────────────────────────
settings_open = False
graphics_quality = 'Yüksek'
advanced_lighting_enabled = True  # Live reflection + AO; hardware DXR/path tracing is not available in Ursina.
saved_player_speed = 6
saved_jump_height = 2.4
menu_time = 0.0

# Menü katmanları: karartma, kenar parıltısı, panel ve dekoratif çizgiler.
settings_root = Entity(parent=camera.ui, enabled=False, z=0)
settings_dim = Entity(parent=settings_root, model='quad', scale=(window.aspect_ratio * 2, 2),
                      color=color.rgba32(2, 3, 10, 225), z=0)
settings_glow = Entity(parent=settings_root, model='quad', scale=(1.02, 1.18),
                       color=color.rgba32(50, 95, 150, 45), z=-0.005)
settings_panel = Entity(parent=settings_root, model='quad', scale=(0.88, 1.12),
                        color=color.rgba32(13, 18, 31, 248), z=-0.01)
settings_panel_inner = Entity(parent=settings_root, model='quad', scale=(0.85, 1.09),
                              color=color.rgba32(20, 27, 43, 248), z=-0.015)
Entity(parent=settings_root, model='quad', scale=(0.84, 0.008), y=0.39,
       color=color.rgb32(75, 190, 255), z=-0.025)
Entity(parent=settings_root, model='quad', scale=(0.84, 0.003), y=-0.535,
       color=color.rgb32(170, 75, 220), z=-0.025)

Text('KARANLIK KRAL', parent=settings_root, y=0.455, scale=0.75,
     origin=(0, 0), color=color.rgb32(115, 205, 255), z=-0.04)
Text('GÖRÜNTÜ / IŞIKLANDIRMA AYARLARI', parent=settings_root, y=0.34, scale=0.92,
     origin=(0, 0), color=color.rgb32(245, 232, 210), z=-0.04)
Text('GÖRSEL KALİTE', parent=settings_root, x=-0.34, y=0.22, scale=0.95,
     origin=(-0.5, 0), color=color.rgb32(185, 202, 225), z=-0.04)

quality_label = Text('SEÇİLİ: YÜKSEK', parent=settings_root, y=0.105, scale=0.95,
                     origin=(0, 0), color=color.rgb32(100, 220, 255), z=-0.04)
ray_label = Text('EKRAN-UZAYI IŞIN İZLEME: KAPALI', parent=settings_root, y=0.035,
                 scale=0.82, origin=(0, 0), color=color.rgb32(255, 180, 95), z=-0.04)
ray_note = Text('Görünmez ışınlar görünen sahne derinliğinde izlenir.',
                parent=settings_root, y=-0.015, scale=0.54, origin=(0, 0),
                color=color.rgb32(165, 178, 198), z=-0.04)
upscaler_label = Text('KARE ARTTIRICI: AÇIK',
                      parent=settings_root, y=-0.145, scale=0.72, origin=(0, 0),
                      color=color.rgb32(110, 245, 165), z=-0.04)
upscaler_note = Text('Düşük çözünürlükten 1080p’ye ölçekler; açıkken görüş dışı modeller geçici gizlenir.',
                     parent=settings_root, y=-0.185, scale=0.48, origin=(0, 0),
                     color=color.rgb32(165, 178, 198), z=-0.04)
upscaler_profile_heading = Text('KARE ARTTIRICI ÇÖZÜNÜRLÜK PROFİLİ',
                               parent=settings_root, y=-0.315, scale=0.62, origin=(0, 0),
                               color=color.rgb32(185, 202, 225), z=-0.04)
upscaler_profile_label = Text('PROFİL: YÜKSEK  |  1280×720 → 1920×1080',
                             parent=settings_root, y=-0.435, scale=0.48, origin=(0, 0),
                             color=color.rgb32(100, 220, 255), z=-0.04)

def _set_shadow_quality(resolution):
    try:
        if resolution <= 0:
            sun._light.setShadowCaster(False)
            sun.shadows = False
        else:
            sun.shadow_map_resolution = Vec2(resolution, resolution)
            sun._light.setShadowCaster(True, int(resolution), int(resolution))
            sun.shadows = True
            sun.update_bounds(shadow_bounds)
    except Exception as shadow_error:
        print('[GÖLGE] Kalite değiştirilemedi:', shadow_error)


def apply_graphics_quality(level):
    global graphics_quality
    graphics_quality = level
    quality_label.text = 'SEÇİLİ: ' + level.upper()

    active_chapter = globals().get('current_ch')
    chapter_fog_color = getattr(active_chapter, 'atmosphere_fog_color', color.rgb32(7, 10, 24))
    chapter_fog_density = getattr(active_chapter, 'atmosphere_fog_density', 0.026)
    if level == 'Düşük':
        scene.fog_density = chapter_fog_density * 0.72
        scene.fog_color = chapter_fog_color
        _set_shadow_quality(512)
    elif level == 'Orta':
        scene.fog_density = chapter_fog_density
        scene.fog_color = chapter_fog_color
        _set_shadow_quality(1024)
    else:
        scene.fog_density = chapter_fog_density * 1.12
        scene.fog_color = chapter_fog_color
        _set_shadow_quality(2048)

    set_postfx_quality(level)
    effects_active = advanced_lighting_enabled and level != 'Düşük' and (RT_PIPELINE_AVAILABLE or reflection_enabled)
    set_reflection_pass(effects_active)
    if RT_PIPELINE_AVAILABLE:
        ray_label.text = 'EKRAN-UZAYI IŞIN İZLEME: ' + ('AÇIK' if effects_active else 'KAPALI')
        ray_label.color = color.rgb32(110, 245, 165) if effects_active else color.rgb32(255, 180, 95)
    else:
        ray_label.text = 'GPU RT SHADER BAŞLATILAMADI'
        ray_label.color = color.rgb32(255, 90, 90)
    for b in quality_buttons:
        selected = b.text.upper() == level.upper()
        b.color = color.rgb32(35, 115, 160) if selected else color.rgb32(35, 43, 62)
        b.highlight_color = color.rgb32(65, 155, 200) if selected else color.rgb32(65, 80, 108)

def apply_ray_tracing(enabled):
    global advanced_lighting_enabled
    advanced_lighting_enabled = bool(enabled)
    active = advanced_lighting_enabled and graphics_quality != 'Düşük' and (RT_PIPELINE_AVAILABLE or reflection_enabled)
    if RT_PIPELINE_AVAILABLE:
        ray_label.text = 'EKRAN-UZAYI IŞIN İZLEME: ' + ('AÇIK' if active else 'KAPALI')
        ray_label.color = color.rgb32(110, 245, 165) if active else color.rgb32(255, 180, 95)
    else:
        ray_label.text = 'GPU RT SHADER BAŞLATILAMADI'
        ray_label.color = color.rgb32(255, 90, 90)
    set_reflection_pass(active)
    # Gölge haritası kalite ayarından bağımsızdır; kapat-aç düğmesi gölgeleri bozmaz.
    _set_shadow_quality({'Düşük': 512, 'Orta': 1024, 'Yüksek': 2048}[graphics_quality])
    set_postfx_quality(graphics_quality)

def apply_frame_upscaler(enabled):
    """Enable or disable spatial upscaling for the active resolution profile."""
    global upscaler_enabled
    requested = bool(enabled)
    if not RT_PIPELINE_AVAILABLE or rt_manager is None or rt_quad is None:
        upscaler_enabled = False
        upscaler_label.text = 'KARE ARTTIRICI: KULLANILAMIYOR (GPU POSTFX KAPALI)'
        upscaler_label.color = color.rgb32(255, 105, 105)
        upscaler_note.text = 'Render hattı başlatılamadığı için ölçekleme uygulanamadı.'
        return

    upscaler_enabled = requested
    try:
        # FilterManager renders internally at the selected profile's source size.
        # The fullscreen output is displayed at that profile's output size.
        rt_manager.forcex = UPSCALE_SOURCE_WIDTH if requested else 0
        rt_manager.forcey = UPSCALE_SOURCE_HEIGHT if requested else 0
        rt_manager.resizeBuffers()
        rt_color_tex.setMinfilter(Texture.FTLinear)
        rt_color_tex.setMagfilter(Texture.FTLinear)
        rt_depth_tex.setMinfilter(Texture.FTNearest)
        rt_depth_tex.setMagfilter(Texture.FTNearest)
        rt_quad.setShaderInput('upscalerEnabled', 1 if requested else 0)
        rt_quad.setShaderInput('upscaleSharpness', 0.42 if requested else 0.0)
        if requested:
            upscaler_label.text = 'KARE ARTTIRICI: AÇIK  |  ' + f'{UPSCALE_SOURCE_WIDTH}×{UPSCALE_SOURCE_HEIGHT} → {UPSCALE_OUTPUT_WIDTH}×{UPSCALE_OUTPUT_HEIGHT}'
        else:
            upscaler_label.text = f'KARE ARTTIRICI: KAPALI  |  ÇIKIŞ {UPSCALE_OUTPUT_WIDTH}×{UPSCALE_OUTPUT_HEIGHT}'
        upscaler_label.color = color.rgb32(110, 245, 165) if requested else color.rgb32(255, 180, 95)
        upscaler_note.text = ('Uzamsal ölçekleme + kenar-duyarlı keskinleştirme; yapay zekâ DLSS değildir.' if requested
                              else 'Profil seçimi saklanır; kapalıyken tam çıkış çözünürlüğü işlenir.')
        update_raytrace_uniforms()
    except Exception as upscale_error:
        upscaler_enabled = False
        upscaler_label.text = 'KARE ARTTIRICI: HATA'
        upscaler_label.color = color.rgb32(255, 90, 90)
        upscaler_note.text = str(upscale_error)[:90]
        print('[KARE ARTTIRICI] Ölçekleme uygulanamadı:', upscale_error)



def apply_upscaler_profile(profile):
    # Apply one of the three requested internal-render/output-resolution profiles.
    global upscaler_profile, UPSCALE_SOURCE_WIDTH, UPSCALE_SOURCE_HEIGHT
    global UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT
    if profile not in UPSCALE_PROFILES:
        return
    upscaler_profile = profile
    (UPSCALE_SOURCE_WIDTH, UPSCALE_SOURCE_HEIGHT,
     UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT) = UPSCALE_PROFILES[profile]

    # All profiles output 1080p; Performance uses a 360p internal render.
    try:
        window.size = (UPSCALE_OUTPUT_WIDTH, UPSCALE_OUTPUT_HEIGHT)
    except Exception as output_size_error:
        print('[KARE ARTTIRICI] Çıkış çözünürlüğü değiştirilemedi:', output_size_error)

    try:
        if RT_PIPELINE_AVAILABLE and rt_manager is not None:
            rt_manager.forcex = UPSCALE_SOURCE_WIDTH if upscaler_enabled else 0
            rt_manager.forcey = UPSCALE_SOURCE_HEIGHT if upscaler_enabled else 0
            rt_manager.resizeBuffers()
        apply_frame_upscaler(upscaler_enabled)
    except Exception as profile_error:
        print('[KARE ARTTIRICI] Profil uygulanamadı:', profile_error)

    upscaler_profile_label.text = (
        f'PROFİL: {profile.upper()}  |  '
        f'{UPSCALE_SOURCE_WIDTH}×{UPSCALE_SOURCE_HEIGHT} → '
        f'{UPSCALE_OUTPUT_WIDTH}×{UPSCALE_OUTPUT_HEIGHT}'
    )
    for button in upscaler_profile_buttons:
        selected = button.text.upper() == profile.upper()
        button.color = color.rgb32(35, 125, 155) if selected else color.rgb32(35, 43, 62)
        button.highlight_color = color.rgb32(70, 175, 205) if selected else color.rgb32(65, 80, 108)
    update_raytrace_uniforms()


def open_settings():
    global settings_open
    if settings_open:
        return
    settings_open = True
    settings_root.enabled = True
    if not hasattr(open_settings, '_saved'):
        open_settings._saved = (player.speed, player.jump_height)
    player.speed = 0
    player.jump_height = 0
    mouse.locked = False
    mouse.visible = True
    # Ursina pauselenince NPC'ler, fiziğin/update'lerin ve oyun zamanlayıcıları durur.
    # Menü entity'leri ignore_paused=True olduğundan tıklanmaya devam eder.
    application.paused = True
    settings_panel.scale = (0.82, 1.04)
    settings_panel.animate_scale((0.88, 1.12), duration=0.16, curve=curve.out_back)

def close_settings():
    global settings_open
    settings_open = False
    settings_root.enabled = False
    if hasattr(open_settings, '_saved'):
        player.speed, player.jump_height = open_settings._saved
        del open_settings._saved
    # Önce oyunu devam ettir, sonra kamera faresini yeniden kilitle.
    application.paused = False
    mouse.locked = True
    mouse.visible = False

def _make_setting_button(label, x, y, action, scale=(0.20, 0.065)):
    b = Button(parent=settings_root, text=label, x=x, y=y, scale=scale,
               color=color.rgb32(35, 43, 62), highlight_color=color.rgb32(65, 115, 150),
               pressed_color=color.rgb32(75, 160, 195), text_color=color.white,
               z=-0.06)
    b.on_click = action
    return b

quality_buttons = []
quality_buttons.append(_make_setting_button('DÜŞÜK', -0.235, 0.155,
                                             lambda: apply_graphics_quality('Düşük')))
quality_buttons.append(_make_setting_button('ORTA', 0, 0.155,
                                             lambda: apply_graphics_quality('Orta')))
quality_buttons.append(_make_setting_button('YÜKSEK', 0.235, 0.155,
                                             lambda: apply_graphics_quality('Yüksek')))
_make_setting_button('IŞIN İZLEME EFEKTLERİ: AÇ / KAPAT', 0, -0.095,
                     lambda: apply_ray_tracing(not advanced_lighting_enabled),
                     scale=(0.48, 0.065))
_make_setting_button('KARE ARTTIRICI: AÇ / KAPAT', 0, -0.245,
                     lambda: apply_frame_upscaler(not upscaler_enabled),
                     scale=(0.42, 0.060))
upscaler_profile_buttons = []
upscaler_profile_buttons.append(_make_setting_button('PERFORMANS', -0.245, -0.375,
                                  lambda: apply_upscaler_profile('Performans'), scale=(0.22, 0.055)))
upscaler_profile_buttons.append(_make_setting_button('DENGELİ', 0, -0.375,
                                  lambda: apply_upscaler_profile('Dengeli'), scale=(0.22, 0.055)))
upscaler_profile_buttons.append(_make_setting_button('YÜKSEK', 0.245, -0.375,
                                  lambda: apply_upscaler_profile('Yüksek'), scale=(0.22, 0.055)))
_make_setting_button('OYUNA DÖN', 0, -0.495, close_settings, scale=(0.28, 0.050))

# Menü arka planına düşük maliyetli, animasyonlu parçacıklar.
menu_sparks = []
for i in range(24):
    spark = Entity(parent=settings_root, model='quad',
                   scale=(random.uniform(0.002, 0.006), random.uniform(0.008, 0.025)),
                   x=random.uniform(-0.82, 0.82), y=random.uniform(-0.52, 0.52),
                   color=color.rgba32(70, 170, 255, random.randint(35, 105)),
                   z=-0.03)
    spark.rotation_z = random.uniform(-25, 25)
    menu_sparks.append((spark, random.uniform(0.2, 0.8)))

# Ursina pauselendiğinde menü elemanları çizilmeye ve giriş almaya devam etsin.
for _menu_entity in settings_root.children:
    _menu_entity.ignore_paused = True

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

# Kare Arttırıcı açıkken görüş dışındaki dekorları geçici olarak gizle.
# Entity'ler yok edilmez: kamera geri döndüğünde tekrar görünürler ve bölüm durumu korunur.
view_culling_enabled = True
view_culling_interval = 0.12
view_culling_timer = 0.0
view_culling_hidden = {}  # id(entity) -> entity; yalnızca bizim gizlediklerimiz


def _is_alive_entity(entity):
    try:
        return entity is not None and not entity.is_empty()
    except Exception:
        return entity is not None


def restore_view_culled_entities():
    """Görüş alanı nedeniyle gizlenen nesneleri geri getir."""
    global view_culling_hidden
    for entity in list(view_culling_hidden.values()):
        if _is_alive_entity(entity):
            try:
                entity.visible = True
            except Exception:
                pass
    view_culling_hidden.clear()


def update_view_culling(force=False):
    """Camera-facing render culling; hides behind/off-view scenery, never destroys it."""
    global view_culling_timer
    if not view_culling_enabled or not upscaler_enabled or settings_open or current_ch is None:
        if view_culling_hidden:
            restore_view_culled_entities()
        return

    view_culling_timer += time.dt
    if not force and view_culling_timer < view_culling_interval:
        return
    view_culling_timer = 0.0

    try:
        # Use horizontal camera direction so looking up/down does not make the ground vanish.
        forward = Vec3(camera.forward.x, 0, camera.forward.z)
        if forward.length() < 0.0001:
            forward = Vec3(player.forward.x, 0, player.forward.z)
        if forward.length() < 0.0001:
            return
        forward = forward.normalized()
        cam_pos = Vec3(camera.world_position.x, 0, camera.world_position.z)
        # 0.12 means a broad forward cone. Anything behind the player is hidden,
        # while near-field floor/platform pieces are preserved to avoid visible holes.
        facing_dot_limit = 0.12
        max_render_distance = 78.0
        near_keep_distance = 4.5

        # No mesh deletion / destruction: hidden items remain interactive and can reappear.
        for entity in list(getattr(current_ch, 'ents', [])):
            if not _is_alive_entity(entity):
                view_culling_hidden.pop(id(entity), None)
                continue

            eid = id(entity)
            try:
                # Very large sky dome and explicitly excluded entities should never be culled.
                if getattr(entity, '_kk_no_view_cull', False):
                    if eid in view_culling_hidden:
                        entity.visible = True
                        view_culling_hidden.pop(eid, None)
                    continue

                scale = entity.scale
                try:
                    scale_x = max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
                except Exception:
                    scale_x = abs(float(scale)) if isinstance(scale, (int, float)) else 1.0
                # Keep giant sky shells, which surround the entire level.
                if scale_x > 100:
                    if eid in view_culling_hidden:
                        entity.visible = True
                        view_culling_hidden.pop(eid, None)
                    continue

                pos = entity.world_position
                delta = Vec3(pos.x - cam_pos.x, 0, pos.z - cam_pos.z)
                distance = delta.length()
                if distance <= near_keep_distance:
                    should_render = True
                elif distance > max_render_distance:
                    should_render = False
                else:
                    direction = delta / max(distance, 0.0001)
                    should_render = forward.dot(direction) >= facing_dot_limit

                if should_render:
                    if eid in view_culling_hidden:
                        # Only restore entities hidden by this routine.
                        entity.visible = True
                        view_culling_hidden.pop(eid, None)
                else:
                    # Skip entities already intentionally hidden by gameplay.
                    if eid not in view_culling_hidden and bool(getattr(entity, 'visible', True)):
                        entity.visible = False
                        view_culling_hidden[eid] = entity
            except Exception:
                # A nonstandard model must not interrupt gameplay's update loop.
                continue
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
    # Ambient particles change palette with the location: fireflies, castle embers, cave sparks.
    palette = {1: color.rgb32(120, 180, 255),
               2: color.rgb32(255, 145, 58),
               3: color.rgb32(255, 85, 28)}[n]
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

def update():
    global menu_time
    # Ek güvenlik: ayarlar açıkken oyun tarafındaki global güncelleme de tamamen durur.
    if settings_open:
        return
    # Bölüm Entity'lerinin update()'i Ursina tarafından zaten çağrılır.
    g.atk_cd = max(0.0, g.atk_cd - time.dt)
    update_warm_light_uniforms()
    update_reflection_camera()
    update_raytrace_uniforms()
    update_view_culling()
    for mote, speed, phase in ambient_motes:
        mote.y += time.dt * speed
        mote.x += math.sin(time.time() * 0.7 + phase) * time.dt * 0.12
        mote.scale = 0.018 + (0.012 * (0.5 + 0.5 * math.sin(time.time() * 1.8 + phase)))
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
    # Enter yalnızca ekrandaki mevcut altyazıyı atlar.
    if key in ('enter', 'numpad enter') and dlg.enabled and not settings_open:
        dlg.hide()
        return
    if key == 'left mouse down' and not settings_open:
        player_attack()

apply_graphics_quality('Yüksek')
apply_ray_tracing(True)
apply_upscaler_profile('Yüksek')
mouse.visible = False
load_chapter(1)
app.run()
