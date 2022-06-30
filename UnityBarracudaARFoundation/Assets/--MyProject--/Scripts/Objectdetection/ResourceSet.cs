using UnityEngine;
using Unity.Barracuda;

namespace ObjectDetection {

[CreateAssetMenu(fileName = "ObjectDetector",
                 menuName = "ScriptableObjects/ObjectDetector Resource Set")]
public sealed class ResourceSet : ScriptableObject
{
    public NNModel model;
    public float[] anchors = new float[12];
    public ComputeShader preprocess;
    public ComputeShader postprocess1;
    public ComputeShader postprocess2;
}

}
