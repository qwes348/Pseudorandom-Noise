using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;

public abstract class Visualization : MonoBehaviour
{
    static int
        positionsId = Shader.PropertyToID("_Positions"),
        normalsId = Shader.PropertyToID("_Normals"),
        configId = Shader.PropertyToID("_Config");

    public enum Shape { Plane, Sphere, Torus }

    static Shapes.ScheduleDelegate[] shapeJobs =
    {
        Shapes.Job<Shapes.Plane>.ScheduleParallel,
        Shapes.Job<Shapes.Sphere>.ScheduleParallel,
        Shapes.Job<Shapes.Torus>.ScheduleParallel
    };

    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-0.5f, 0.5f)]
    float displacement = 0.1f;   

    [SerializeField]
    Shape shape;

    [SerializeField, Range(0.1f, 10f)]
    float instanceScale = 2f;

    NativeArray<float3x4> positions, normals;

    ComputeBuffer positionsBuffer, normalsBuffer;

    MaterialPropertyBlock propertyBlock;

    bool isDirty;

    Bounds bounds;

    private void OnEnable()
    {
        isDirty = true;
        int length = resolution * resolution;
        length = length / 4 + (length & 1);
        positions = new NativeArray<float3x4>(length, Allocator.Persistent);
        normals = new NativeArray<float3x4>(length, Allocator.Persistent);
        positionsBuffer = new ComputeBuffer(length * 4, 3 * 4);
        normalsBuffer = new ComputeBuffer(length * 4, 3 * 4);

        propertyBlock ??= new MaterialPropertyBlock();
        EnableVisualization(length, propertyBlock);
        propertyBlock.SetBuffer(positionsId, positionsBuffer);
        propertyBlock.SetBuffer(normalsId, normalsBuffer);
        propertyBlock.SetVector(configId, new Vector4(resolution, instanceScale / resolution, displacement));
    }

    private void OnDisable()
    {
        positions.Dispose();
        normals.Dispose();

        positionsBuffer.Release();
        normalsBuffer.Release();

        positionsBuffer = null;
        normalsBuffer = null;

        DisableVisualization();
    }

    private void OnValidate()
    {
        if (positionsBuffer != null && enabled)
        {
            OnDisable();
            OnEnable();
        }
    }

    private void Update()
    {
        if(isDirty || transform.hasChanged)
        {
            isDirty = false;
            transform.hasChanged = false;

            UpdateVisualization(positions, resolution, shapeJobs[(int)shape](positions, normals, resolution, transform.localToWorldMatrix, default));

            positionsBuffer.SetData(positions.Reinterpret<float3>(3 * 4 * 4));
            normalsBuffer.SetData(normals.Reinterpret<float3>(3 * 4 * 4));

            bounds = new Bounds(
                transform.position,
                float3(2f * cmax(abs(transform.lossyScale)) + displacement)
                );
        }

        Graphics.DrawMeshInstancedProcedural(instanceMesh, 0, material, bounds, resolution * resolution, propertyBlock);
    }

    protected abstract void EnableVisualization(int dataLength, MaterialPropertyBlock propertyBlock);

    protected abstract void DisableVisualization();

    protected abstract void UpdateVisualization(NativeArray<float3x4> positions, int resolution, JobHandle handle);

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
