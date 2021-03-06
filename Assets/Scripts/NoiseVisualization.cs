using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;
using static Noise;

public class NoiseVisualization : Visualization
{
    static int
        noiseId = Shader.PropertyToID("_Noise");

    static ScheduleDelegate[,] noiseJobs = 
    {
        {
            Job<Lattice1D<LatticeNormal, Perlin>>.ScheduleParallel,
            Job<Lattice1D<LatticeTiling, Perlin>>.ScheduleParallel,
            Job<Lattice2D<LatticeNormal, Perlin>>.ScheduleParallel,
            Job<Lattice2D<LatticeTiling, Perlin>>.ScheduleParallel,
            Job<Lattice3D<LatticeNormal, Perlin>>.ScheduleParallel,
            Job<Lattice3D<LatticeTiling, Perlin>>.ScheduleParallel
        },
        {
            Job<Lattice1D<LatticeNormal, Turbulence<Perlin>>>.ScheduleParallel,
            Job<Lattice1D<LatticeTiling, Turbulence<Perlin>>>.ScheduleParallel,
            Job<Lattice2D<LatticeNormal, Turbulence<Perlin>>>.ScheduleParallel,
            Job<Lattice2D<LatticeTiling, Turbulence<Perlin>>>.ScheduleParallel,
            Job<Lattice3D<LatticeNormal, Turbulence<Perlin>>>.ScheduleParallel,
            Job<Lattice3D<LatticeTiling, Turbulence<Perlin>>>.ScheduleParallel
        },
        {
            Job<Lattice1D<LatticeNormal, Value>>.ScheduleParallel,
            Job<Lattice1D<LatticeTiling, Value>>.ScheduleParallel,
            Job<Lattice2D<LatticeNormal, Value>>.ScheduleParallel,
            Job<Lattice2D<LatticeTiling, Value>>.ScheduleParallel,
            Job<Lattice3D<LatticeNormal, Value>>.ScheduleParallel,
            Job<Lattice3D<LatticeTiling, Value>>.ScheduleParallel
        },
        {
            Job<Lattice1D<LatticeNormal, Turbulence<Value>>>.ScheduleParallel,
            Job<Lattice1D<LatticeTiling, Turbulence<Value>>>.ScheduleParallel,
            Job<Lattice2D<LatticeNormal, Turbulence<Value>>>.ScheduleParallel,
            Job<Lattice2D<LatticeTiling, Turbulence<Value>>>.ScheduleParallel,
            Job<Lattice3D<LatticeNormal, Turbulence<Value>>>.ScheduleParallel,
            Job<Lattice3D<LatticeTiling, Turbulence<Value>>>.ScheduleParallel
        }
    };

    [SerializeField, Range(1, 3)]
    int dimensions = 3;

    [SerializeField]
    Settings noiseSettings = Settings.Default;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS
    {
        scale = 8f
    };

    public enum NoiseType { Perlin, PerlinTurbulence, Value, ValueTurbulence }
    
    [SerializeField]
    NoiseType type;

    [SerializeField]
    bool tiling;

    NativeArray<float4> noise;

    ComputeBuffer noiseBuffer;

    protected override void EnableVisualization(int dataLength, MaterialPropertyBlock propertyBlock)
    {
        noise = new NativeArray<float4>(dataLength, Allocator.Persistent);
        noiseBuffer = new ComputeBuffer(dataLength * 4, 4);

        propertyBlock.SetBuffer(noiseId, noiseBuffer);                
    }

    protected override void DisableVisualization()
    {
        noise.Dispose();
        noiseBuffer.Release();
        noiseBuffer= null;
    }

    protected override void UpdateVisualization(NativeArray<float3x4> positions, int resolution, JobHandle handle)
    {
        noiseJobs[(int)type, 2 * dimensions - (tiling ? 1 : 2)](positions, noise, noiseSettings, domain, resolution, handle).Complete();
        noiseBuffer.SetData(noise.Reinterpret<float>(4 * 4));
    }

    struct HashJob : IJobFor
    {
        [ReadOnly]
        public NativeArray<float3x4> positions;

        [WriteOnly]
        public NativeArray<uint4> hashes;

        public SmallXXHash4 hash;

        public float3x4 domainTRS;

        float4x3 TransformPositions(float3x4 trs, float4x3 p) => float4x3(
            trs.c0.x * p.c0 + trs.c1.x * p.c1 + trs.c2.x * p.c2 + trs.c3.x,
            trs.c0.y * p.c0 + trs.c1.y * p.c1 + trs.c2.y * p.c2 + trs.c3.y,
            trs.c0.z * p.c0 + trs.c1.z * p.c1 + trs.c2.z * p.c2 + trs.c3.z
        );

        public void Execute(int i)
        {
            float4x3 p = TransformPositions(domainTRS, transpose(positions[i]));

            int4 u = (int4)floor(p.c0);
            int4 v = (int4)floor(p.c1);
            int4 w = (int4)floor(p.c2);

            hashes[i] = hash.Eat(u).Eat(v).Eat(w);
        }
    }
}
