using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
[CreateAssetMenu(fileName = "ComputeShaderTest", menuName = "Test/ComputeShaderTest")]
public class ComputeShaderTest : MonoBehaviour
{
    public int size;
    public ComputeShader compute;
    public RawImage Source;
    public RawImage Result;
}
