using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public interface IObjectDetector
{
    IEnumerator ProcessImage(Texture sourceTexture, System.Action<Tensor> result);
    void Start();
    IEnumerator Detect(Tensor tex, float threshold, System.Action<IList<Detection>> result);
    bool Detecting { get; }
    IList<Detection> Detections { get; }
    void Destroy();

}
