Shader "VolumeRenderer/VolumeRaymarch"
{

    Properties
    {
        _Volume         ("Volume Texture", 3D)       = "white" {}
        _VolumeDimensions("Volume Dimensions (XYZ)", Vector) = (1, 1, 1, 0)
        _StepSize       ("Raymarch Step Size",  Range(0.001, 0.05)) = 0.01
        _Threshold      ("Alpha Threshold",     Range(0.0, 1.0))    = 0.1
        _Offset         ("Texture UV Offset",   Vector) = (0, 0, 0)
    }

    SubShader
    {
        // Transparent rendering — required for volume blending
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Front   // Render back faces so we can enter the cube from the front
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma target   4.5
            #pragma vertex   vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            #include "UnityStandardUtils.cginc"

            sampler3D _Volume;
            float4    _VolumeDimensions;
            float     _StepSize;
            float     _Threshold;
            float4    _Offset;
            int _MaxSteps;
            sampler2D _TransferLUT;

            //Shade params
            int _IsShaded;
            float4 _LightPosition;
            float4 _LightColor;
            float  _LightIntensity;
            float  _Metallic;
            float _Specular;
            float _AmbientPercent;
            sampler3D _GradientVolume;
            //End of shade params

            //SSS params
            float _ScatteringCoeff;
            int _ScatteringSteps;
            int _SSSisOn;
            float4 _ScatterColor;
            sampler3D _LightVolume;
            int _ScatteringSamples;
            float _ScatteringRadius;
            float _Anisotropy;
            //End SSS params

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 clipPos   : SV_POSITION;
                float3 objectPos : TEXCOORD0;   // Vertex in object-space [−0.5, 0.5]³
                float3 worldPos  : TEXCOORD1;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.clipPos   = UnityObjectToClipPos(v.vertex);
                o.objectPos = v.vertex.xyz;
                o.worldPos  = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            //For computing the probability of light scattering towards the camera
            float henyenGreenstein(float cosTheta, float g) {
                float g2 = g * g;
                float phase = (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
                return phase;
            }

            // ---- AABB ray–box intersection in object space ----
            // Returns (tEntry, tExit); tEntry >= tExit means no hit.
            float2 intersectBox(float3 rayOrigin, float3 rayDir)
            {
                float3 boxMin = float3(-0.5, -0.5, -0.5);
                float3 boxMax = float3( 0.5,  0.5,  0.5);

                float3 invDir = 1.0 / rayDir;
                float3 t0 = (boxMin - rayOrigin) * invDir;
                float3 t1 = (boxMax - rayOrigin) * invDir;

                float3 tMin = min(t0, t1);
                float3 tMax = max(t0, t1);

                float tEntry = max(max(tMin.x, tMin.y), tMin.z);
                float tExit  = min(min(tMax.x, tMax.y), tMax.z);
                return float2(tEntry, tExit);
            }

            float3 computeTransmittance(float3 uvw, float3 lightDirObj, int steps, float3 mapped)
            {
                // Find how far we can march before leaving the box
                float2 tHit = intersectBox(uvw - 0.5, lightDirObj); // uvw back to [-0.5,0.5] space
                float marchDist = max(tHit.y, 0.0);

                // Divide that distance evenly across steps
                float dt = marchDist / max(steps, 1);
                float3 stepVec = lightDirObj * dt;

                float absorption = 0.0;
                float3 pos = uvw;

                for (int i = 0; i < steps; i++)
                {
                    pos += stepVec;
                    if (any(pos < 0.0) || any(pos > 1.0)) break;
                    absorption += tex3Dlod(_Volume, float4(pos, 0)).r * dt;
                }

                // Now _ScatteringCoeff directly scales optical depth — coefficient has real effect
                float transmittanceScalar = exp(-absorption * _ScatteringCoeff);
                return transmittanceScalar * mapped.rgb;
            }


            float3 sphereSamplePoint(float3 center, int i, int total, float radius)
            {
                float golden = 2.399963;
                float phi = golden * i;
                float cosTheta = 1.0 - 2.0 * (i + 0.5) / total;
                float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

                float3 dir = float3(
                    sinTheta * cos(phi),
                    sinTheta * sin(phi),
                    cosTheta
                    );
                return center + dir * radius;
            }

            // This essentially does a montecarlo estimation of the indirect light contribution 
            float3 computeSSS(float3 uvw, float3 lightDirObj, float3 mappedRGB, float3 viewDirObj)
            {
                // Direct — sample light from current point
                float3 direct = computeTransmittance(uvw, lightDirObj, _ScatteringSteps, mappedRGB);

                // Indirect — sample scattered points then to light
                float3 indirect = float3(0, 0, 0);
                float3 indirectSum = float3(0, 0, 0);
                int hit = 0;

                float spherePDF = 1.0 / (4.0 * 3.14159265 * _ScatteringRadius * _ScatteringRadius); // to divide by for MIS

                for (int i = 0; i < _ScatteringSamples; i++)
                {
                    float3 sampleUVW = sphereSamplePoint(uvw, i, _ScatteringSamples, _ScatteringRadius);

                    if (any(sampleUVW < 0.0) || any(sampleUVW > 1.0)) continue;

                    float density = tex3Dlod(_Volume, float4(sampleUVW, 0)).r;
                    if (density < _Threshold) continue;

                    float3 indirectSample = computeTransmittance(sampleUVW, lightDirObj, _ScatteringSteps, mappedRGB);
                    indirectSum += indirectSample / spherePDF;
                    hit++;
                }

                if (hit == 0) return direct;
                else indirect = indirectSum / hit;

                float phase = henyenGreenstein(dot(-lightDirObj, viewDirObj), _Anisotropy);
                float isotropicPhase = 1.0 / (4.0 * 3.14159265);
                float phaseWeight = phase / isotropicPhase;
                return ((direct + indirect) / 2.0) * phaseWeight;
            }

            //This uses cook-torrence microfacet with the ggx distribution 
            float3 Shade(float3 N, float3 vertPos, float3 V, float3 texColor, float3 transmittance, float4 mapped)
            {
                float3 totalColor = float3(0, 0, 0);

                float3 L = normalize(_LightPosition.xyz - vertPos);
                float3 I = _LightColor.rgb * _LightIntensity;
                if (_SSSisOn == 1) {
                    I = (transmittance * mapped.rgb) * _LightIntensity;
                }
                float3 H = normalize(L + V);
                //float _WrapAmount = 0.3;  // 0 = no wrap, 1 = full hemisphere
                //float NDotL = max((dot(N, L) + _WrapAmount) / (1.0 + _WrapAmount), 0.0);
                float NDotL = max(dot(N, L), 0.0);
                float3 F0 = float3(0, 0, 0);

                if (NDotL > 0.0)
                {
                    float  NDotV = max(dot(N, V), 0.0);
                    float  NDotH = max(dot(N, H), 0.0);
                    float  VDotH = max(dot(V, H), 0.0);

                    // Fresnel base reflectivity — lerp between dielectric (0.04) and metal (texColor)
                    F0 = lerp(float3(0.04, 0.04, 0.04), texColor, _Metallic);

                    // Fresnel (Schlick)
                    float3 F = F0 + (1.0 - F0) * pow(1.0 - VDotH, 5.0);

                    // NDF — GGX/Trowbridge-Reitz
                    float alpha = _Specular * _Specular;
                    float alpha2 = alpha * alpha;
                    float denom = (NDotH * NDotH) * (alpha2 - 1.0) + 1.0;
                    float D = alpha2 / (3.14159265 * denom * denom);

                    // Geometry — Smith's method
                    float k = alpha2 / 2;
                    float3 G = (NDotV / (NDotV * (1.0 - k) + k)) * (NDotL / (NDotL * (1.0 - k) + k));
                    // Specular + diffuse
                    float  specDenom = max(4.0 * NDotV * NDotL, 0.001);
                    float3 spec = (D * G * F) / specDenom;
                    float3 kd = (1.0 - F) * (1.0 - _Metallic);
                    float3 diff = kd * texColor / 3.14159265;

                    totalColor += (diff + spec) * I * NDotL;
                }
                // Reflect view direction around normal for env map lookup
                float3 reflDir = reflect(-V, N);  // V is already world space, N is world space normal

                // Sample the reflection probe
                // Use roughness to select mip level — rough surfaces sample blurrier mips
                float mipLevel = _Specular * UNITY_SPECCUBE_LOD_STEPS;
                float4 envSample = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, reflDir, mipLevel);

                // Unity stores env maps in HDR, decode it first
                float3 envColor = DecodeHDR(envSample, unity_SpecCube0_HDR);

                // Weight env reflection by Fresnel — grazing angles reflect more
                float3 envFresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
                float3 envSpecular = envColor * envFresnel * (1.0 - _Specular) * texColor;  // rougher = less env reflection
                totalColor += envSpecular;

                float3 ambientTerm = (_LightColor * _AmbientPercent);
                return totalColor + ambientTerm;
            }

            

            float4 frag(v2f i) : SV_Target
            {
                //Ray setup in object space
                float3 camPosObj = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1.0)).xyz;
                float3 rayDir    = normalize(i.objectPos - camPosObj);

                float2 tHit = intersectBox(camPosObj, rayDir);
                if (tHit.x >= tHit.y) discard;

                float tEntry = max(tHit.x, 0.0);
                float tExit  = tHit.y;

                //Raymarch accumulation 
                float4 accum   = float4(0, 0, 0, 0);
                float  t       = tEntry;
                int    maxSteps = _MaxSteps;

                // Trying to get rid of aliasing
                float jitter = frac(sin(dot(i.clipPos.xy, float2(127.1, 311.7))) * 43758.5453);
                t = tEntry + jitter * _StepSize;

                float3 dims = _VolumeDimensions.xyz;
                float3 halfVoxel = 0.5 / dims;

                for (int s = 0; s < maxSteps && t < tExit; s++)
                {
                    float3 pos = camPosObj + t * rayDir;         // object space [−0.5, 0.5]
                    float3 uvw = (pos + 0.5);
                    uvw = halfVoxel + uvw * (1.0 - 2.0 * halfVoxel); // remap to voxel centers
                    uvw += _Offset.xyz;
                    float density = tex3Dlod(_Volume, float4(uvw, 0)).r;

                    // STUB: visualise density as greyscale with threshold cut
                    if (density > _Threshold)
                    {
                        float4 mapped = tex2Dlod(_TransferLUT, float4(density, 0.5, 0, 0));
                        mapped.a *= 100.0 * _StepSize;
                        float3 transmittance = float3(0, 0, 0);
                        float3 worldPos = mul(unity_ObjectToWorld, float4(pos, 1.0)).xyz;
                        float3 viewDir = normalize(camPosObj - pos);
                        if (_SSSisOn == 1) {
                            float3 lightDir = normalize(_LightPosition.xyz - worldPos);
                            float3 lightDirObj = normalize(mul((float3x3)unity_WorldToObject, lightDir));
                            transmittance = computeSSS(uvw, lightDirObj, mapped.rgb, viewDir);  // was computeTransmittance
                        }
                        if (_IsShaded == 1) {
                            float4 gradSample = tex3Dlod(_GradientVolume, float4(uvw, 0));
                            float3 normal = normalize(gradSample.rgb * 2.0 - 1.0); // remmap 0-1 back to -1-1
                            float3 normalWorld = normalize(mul(-normal, (float3x3)unity_WorldToObject)); //forgot to transform to world earlier, whoops
                            float3 viewDirWorld = normalize(mul((float3x3)unity_ObjectToWorld, viewDir));
                            float3 litColor = Shade(normalWorld, worldPos, viewDirWorld, mapped.rgb, transmittance, mapped);
                            if (_SSSisOn == 1) {
                                // transmittance is how much light survives — tint it and add as backlit glow
                                // scale only by density so empty voxels don't contribute
                                float3 sss = transmittance * mapped.rgb;
                                litColor = litColor + sss * _LightIntensity * _LightColor.rgb;
                            }
                            // Skip shadingWeight entirely — it's fighting SSS and isn't needed
                            mapped.rgb = litColor;
                        }
                        accum.rgb += (1.0 - accum.a) * mapped.a * mapped.rgb;
                        accum.a += (1.0 - accum.a) * mapped.a;
                    }

                    if (accum.a >= 0.99) break;  // early termination
                    t += _StepSize;
                }

                // Discard fully transparent fragments
                if (accum.a < 0.001) discard;
                return accum;
            }
            ENDCG
        }
    }
}
